use core::mem::size_of;
use core::sync::atomic::{AtomicUsize, Ordering, spin_loop_hint};
use crate::syncbuf::Syncbuf;

const FIRST_CHUNK_SIZE: usize = 16;

/// Growable, thread-safe buffer that allows adding new elements
/// without invalidating shared references.
///
/// # Notes
///
/// `Synclist` is dynamically growable.  This is done using a chunking scheme
/// adapted from the [`appendlist` crate](https://docs.rs/appendlist/latest/appendlist/).
/// This approach adds 2 sources of overhead versus [`Syncbuf`](super::syncbuf::Syncbuf):
/// - One additional indirection on indexing operations.
/// - It allocates all of its "chunks" upfront, so creating an empty
///   `Synclist` immediately allocates 448 bytes on 32-bit
///   platforms, or 1920 bytes on 64-bit platforms.
///
/// `Synclist` has the same size constraint as [`Vec`]; attempting to add more
/// than `isize::MAX` elements will panic.

#[derive(Debug)]
pub struct Synclist<T> {
    buf: Syncbuf<Syncbuf<T>>,
    // the chunk idx where new elements should be pushed
    // if there is no chunk at this index, that means another thread is currently adding it
    last_chunk: AtomicUsize,
}

// The next 4 functions were adapted from danieldulaney's `appendlist` crate
#[inline]
const fn floor_log2(x: usize) -> usize {
    const BITS_PER_BYTE: usize = 8;
    BITS_PER_BYTE * size_of::<usize>() - (x.leading_zeros() as usize) - 1
}

impl<T> Synclist<T> {
    /// Returns what the size of a specific chunk is
    const fn chunk_size(chunk_idx: usize) -> usize {
        FIRST_CHUNK_SIZE << chunk_idx
    }

    /// Returns the index of the full list at which the specified chunk begins
    const fn chunk_start(chunk_idx: usize) -> usize {
        Synclist::<T>::chunk_size(chunk_idx) - FIRST_CHUNK_SIZE
    }

    /// For an index in the list, return which chunk it will be within
    const fn index_chunk(idx: usize) -> usize {
        floor_log2(idx + FIRST_CHUNK_SIZE) - floor_log2(FIRST_CHUNK_SIZE)
    }

    /// The number of chunks a `Synclist` needs to have to be able to contain
    /// isize::MAX elements
    const fn num_chunks() -> usize {
        Synclist::<T>::index_chunk(isize::MAX as usize - 1) + 1
    }

    /// Creates a `Synclist` that can hold at least `capacity` elements without
    /// new allocations.
    pub fn with_capacity(mut capacity: usize) -> Synclist<T> {
        let buf = Syncbuf::with_capacity(Synclist::<T>::num_chunks());
        // TODO: investigate if an optimization is possible here to have the first few chunks
        // contiguous within a single allocation when `capacity` > FIRST_CHUNK_SIZE
        capacity = core::cmp::max(capacity, FIRST_CHUNK_SIZE);
        let num_chunks_initial = Synclist::<T>::index_chunk(capacity-1) + 1;
        for i in 0..num_chunks_initial {
            let chunk = Syncbuf::with_capacity(Synclist::<T>::chunk_size(i));
            match buf.push(chunk) {
                Ok(_) => (),
                _ => unreachable!("pushing past allowable chunk size should not be possible"),
            };
        }
        Synclist {
            buf,
            last_chunk: AtomicUsize::new(0),
        }
    }

    /// Creates a `Synclist` with default capacity (16 for now)
    pub fn new() -> Synclist<T> {
        Synclist::with_capacity(FIRST_CHUNK_SIZE)
    }

    /// Helper function to get last chunk and its index because it could involve waiting
    fn get_last_chunk(&self) -> (usize, &Syncbuf<T>) {
        loop {
            let last_idx = self.last_chunk.load(Ordering::SeqCst);
            match self.buf.get(last_idx) {
                Some(chunk) => break (last_idx, chunk),
                None => spin_loop_hint(),
            }
        }
    }

    /// Gets a reference to an element.
    pub fn get(&self, idx: usize) -> Option<&T> {
        let chunk_idx = Synclist::<T>::index_chunk(idx);
        let elem_idx = idx - Synclist::<T>::chunk_start(chunk_idx);
        match self.buf.get(chunk_idx) {
            None => None,
            Some(chunk) => chunk.get(elem_idx),
        }
    }

    /// Returns the number of elements in the list.
    pub fn len(&self) -> usize {
        let (last_idx, last_chunk) = self.get_last_chunk();
        Synclist::<T>::chunk_start(last_idx) + last_chunk.len()
    }

    /// Returns the number of elements the list can hold without new allocations.
    pub fn capacity(&self) -> usize {
        let (last_idx, last_chunk) = self.get_last_chunk();
        Synclist::<T>::chunk_start(last_idx) + last_chunk.capacity()
    }

    /// Helper to add a new chunk with one element.
    /// Note: acquire the pseudo-lock before calling this
    /// or the list may become sparse (this would not be UB, but would be bad)
    fn push_chunk_with_capacity_and_elem(&self, capacity: usize, elem: T) -> (usize, usize) {
        if capacity >= (isize::MAX as usize >> 1) {
            panic!("Synclist cannot hold more than isize::MAX elements");
        }
        let new_chunk = Syncbuf::<T>::with_capacity(capacity);
        let elem_idx = match new_chunk.push(elem) {
            Err(_) => unreachable!("brand new Syncbuf should have capacity"),
            Ok(i) => i,
        };
        let chunk_idx = match self.buf.push(new_chunk) {
            Err(_) => unreachable!("Synclist cannot hold more than isize::MAX elements"),
            Ok(i) => i,
        };
        (chunk_idx, elem_idx)
    }

    /// Pushes an element, returning its index.
    pub fn push(&self, elem: T) -> usize {
        let (chunk_idx, elem_idx) = {
            let (chunk_idx, chunk) = self.get_last_chunk();
            // try to push to this last buffer
            match chunk.push(elem) {
                Ok(i) => (chunk_idx, i),
                // if we fail, that means this chunk is full
                Err(elem) => {
                    // Attempt to acquire the pseudo-lock on adding a chunk
                    match self.last_chunk.compare_exchange(chunk_idx, chunk_idx+1, Ordering::SeqCst, Ordering::SeqCst) {
                        Ok(_) => {
                            // during tests, slow down the critical path here to force more thread contention
                            #[cfg(test)]
                            {
                                extern crate std;
                                std::thread::sleep(std::time::Duration::from_millis(1000));
                            }
                            match self.buf.get(chunk_idx+1) {
                                Some(chunk) => {
                                    // we already had another chunk allocated here
                                    match chunk.push(elem) {
                                        // this is the rare case discussed below
                                        Err(elem) => return self.push(elem),
                                        Ok(i) => (chunk_idx+1, i),
                                    }
                                },
                                None => {
                                    // we need to do the push
                                    self.push_chunk_with_capacity_and_elem(
                                        Synclist::<T>::chunk_size(chunk_idx+1),
                                        elem,
                                    )
                                }
                            }
                        },
                        Err(_) => {
                            // this means another thread already added a chunk
                            // so let's try to use it
                            let (new_last_idx, new_last_chunk) = self.get_last_chunk();
                            match new_last_chunk.push(elem) {
                                Ok(i) => (new_last_idx, i),
                                Err(elem) => {
                                    // Rare case where a concurrent chunk was pushed, but
                                    // somehow was completely filled before this thread got to it.
                                    // This is so unlikely that the simplest fix is to try again
                                    // recursively
                                    return self.push(elem)
                                }
                            }
                        }
                    }
                }
            }
        };
        Synclist::<T>::chunk_start(chunk_idx) + elem_idx
    }
}

impl<T> core::ops::Index<usize> for Synclist<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("Index out-of-bounds")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::string::{String, ToString};
    use std::vec::Vec;

    #[test]
    fn index_fns() {
        assert_eq!(Synclist::<i32>::chunk_size(0), FIRST_CHUNK_SIZE);
        assert_eq!(Synclist::<i32>::chunk_size(1), FIRST_CHUNK_SIZE*2);
        assert_eq!(Synclist::<i32>::chunk_size(2), FIRST_CHUNK_SIZE*4);
        let mut index = 0;

        for chunk in 0..20 {
            // Each chunk starts just after the previous one ends
            assert_eq!(Synclist::<i32>::chunk_start(chunk), index);
            index += Synclist::<i32>::chunk_size(chunk);
        }

        for index in 0..1_000_000 {
            let chunk_id = Synclist::<i32>::index_chunk(index);

            // Each index happens after its chunk start and before its chunk end
            assert!(index >= Synclist::<i32>::chunk_start(chunk_id));
            assert!(index < Synclist::<i32>::chunk_start(chunk_id) + Synclist::<i32>::chunk_size(chunk_id));
        }
    }

    #[test]
    fn simple() {
        let list = Synclist::<String>::new();
        list.push("foo".to_string());
        list.push("bar".to_string());
        assert_eq!(list.len(), 2);
        assert_eq!(list[1], "bar");
        assert_eq!(list.get(2), None);
    }

    #[test]
    fn refs_not_invalidated() {
        let list = Synclist::<String>::new();
        list.push("foo".to_string());
        let foo_ref = &list[0];
        list.push("bar".to_string());
        assert_eq!(foo_ref, "foo");
    }

    #[test]
    fn many_threads() {
        // yes, this is virtually identical to the same test from `Syncbuf`
        use std::sync::Arc;
        use std::thread::{sleep, JoinHandle};
        use std::time::Duration;

        const THREADS: usize = 10;
        const PUSHES: usize = 100;

        let buf: Synclist<i32> = Synclist::with_capacity(1000);
        std::println!("Initialized Synclist: {:?}", buf);
        std::println!("last chunk: {:?}", buf.get_last_chunk());
        let buf_arc = Arc::new(buf);
        let mut children: Vec<JoinHandle<_>> = Vec::new();
        for i in 0..THREADS as i32 {
            let buf_arc = Arc::clone(&buf_arc);
            children.push(std::thread::spawn(move || {
                for _ in 0..PUSHES {
                    sleep(Duration::from_millis(2));
                    let idx = buf_arc.push(i);
                    let i_ref = buf_arc.get(idx).unwrap();
                    sleep(Duration::from_millis(2));
                    // our reference still works after concurrent pushes
                    assert_eq!(*i_ref, i);
                }
            }));
        }
        for handle in children {
            handle.join().unwrap();
        }
        std::println!("{:?}", buf_arc);
        assert_eq!(buf_arc.len(), THREADS * PUSHES);
    }


}
