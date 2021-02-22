extern crate alloc;
use alloc::vec::Vec;
use core::mem::{ManuallyDrop, size_of};
use core::sync::atomic::{AtomicUsize, AtomicPtr, Ordering, spin_loop_hint};

/// Fixed-size, thread-safe buffer that allows adding new
/// elements without invalidating shared references
///
/// # Examples
///
/// ```rust
/// # use std::thread;
/// # use std::sync::Arc;
/// # use syncbuf::Syncbuf;
/// // Syncbuf has an explicit, fixed capacity; no reallocations
/// let buf: Syncbuf<String> = Syncbuf::with_capacity(4);
///
/// // Push returns the index of the pushed element
/// assert_eq!(buf.push("foo".to_owned()), Ok(0));
///
/// // References can be kept through modifications
/// let my_ref = &buf[0];
/// assert_eq!(buf.push("bar".to_owned()), Ok(1));
/// assert_eq!(my_ref, "foo");
///
/// // Buffer can be modified and read in parallel
/// let buf_arc = Arc::new(buf);
/// let my_ref = &buf_arc[0];
/// let send_arc = buf_arc.clone();
/// thread::spawn(move || {
///     send_arc.push("bat".to_owned())
/// }).join();
/// assert_eq!(my_ref, "foo");
/// assert_eq!(&buf_arc[2], "bat");
/// ```

pub struct Syncbuf<T> {
    // INVARIANTS:
    // - indices before `len` must have valid elements in them and never be mutated
    // - indices after `working_len` are uninitialized
    // - indices between `len` and `working_len` are in the process of being
    //   written and must not be accessed outside of the writing thread
    pub(crate) capacity: usize,
    pub(crate) len: AtomicUsize,
    pub(crate) working_len: AtomicUsize,
    pub(crate) buf: AtomicPtr<T>,
}

impl<T> Syncbuf<T> {
    /// Allocates a new `Syncbuf` with a fixed capacity.
    pub fn with_capacity(capacity: usize) -> Syncbuf<T> {
        let v: Vec<T> = Vec::with_capacity(capacity);
        Syncbuf::from(v)
    }

    /// Pushes an element at the end of the buffer, returning its index
    ///
    /// Returns `Ok(index)` if the push was successful.
    /// Returns `Err(elem)` if the buffer was already full.
    pub fn push(&self, elem: T) -> Result<usize, T> {
        // We only ever write to the buffer:
        //  - After incrementing working_len
        //  - At the index working_len - 1
        let idx = self.working_len.fetch_add(1, Ordering::SeqCst);
        if idx >= self.capacity {
            return Err(elem);
        }

        // during tests, slow down the critical path here to force more thread contention
        #[cfg(test)]
        {
            extern crate std;
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // SAFETY: we have already checked that `idx` is within the
        // allocation and we will never write to the same `idx` twice
        unsafe {
            let location = self.buf.load(Ordering::Relaxed).add(idx);
            location.write(elem);
        }

        // It is possible the element we just wrote was more than 1 past `len`
        // if this is the case, we have to wait until any concurrent pushes finish
        while self.len.compare_and_swap(idx, idx + 1, Ordering::SeqCst) != idx {
            spin_loop_hint();
        }
        Ok(idx)
    }

    /// Gets a reference to an element.
    ///
    /// Returns `None` if the index is out of bounds
    pub fn get(&self, index: usize) -> Option<&T> {
        let len = self.len.load(Ordering::SeqCst);
        if index >= len {
            None
        } else {
            // SAFETY: `idx` < `len` < `capacity` so `elem_ptr` is non-null
            // We never give out `&mut`, so this shared reference is safe
            let elem_ref = unsafe {
                let elem_ptr = self.buf.load(Ordering::Relaxed).add(index);
                elem_ptr.as_ref()
            };
            elem_ref
        }
    }

    /// Returns the last element in the buffer, if there is one.
    pub fn last(&self) -> Option<&T> {
        self.last_item().map(|(_,r)| r)
    }

    /// Returns the last element and its index, if there is one.
    pub fn last_item(&self) -> Option<(usize, &T)> {
        match self.len.load(Ordering::SeqCst) {
            0 => None,
            i => self.get(i-1).map(|e| (i-1, e)),
        }
    }

    /// Returns the number of elements in the buffer
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Returns the number of elements the buffer can hold
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns an `Iterator` over the contents of the buffer
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            idx: 0,
            orig: self,
        }
    }
}

impl<T> From<Vec<T>> for Syncbuf<T> {
    /// Panics if `T` is zero-sized
    fn from(v: Vec<T>) -> Self {
        if size_of::<T>() * v.capacity() == 0 {
            panic!("Syncbuf does not support ZSTs");
        }
        let mut v = ManuallyDrop::new(v);
        let vec_buf = v.as_mut_ptr();
        let buf: AtomicPtr<T> = AtomicPtr::new(vec_buf);
        Syncbuf {
            capacity: v.capacity(),
            buf,
            len: AtomicUsize::new(v.len()),
            working_len: AtomicUsize::new(v.len()),
        }
    }
}

/// Iterator over a `Syncbuf`'s contents
pub struct Iter<'i, T: 'i> {
    // We don't just build a `std::slice::Iter` from the underlying buffer
    // because that would require setting the bounds when the iterator is
    // created, making it unable to observe concurrent pushes
    orig: &'i Syncbuf<T>,
    idx: usize,
}

impl<'i, T> Iterator for Iter<'i, T> {
    type Item = &'i T;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.orig.get(self.idx);
        self.idx += 1;
        ret
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.orig.len() - self.idx, Some(self.orig.capacity - self.idx))
    }
}

impl<T> core::ops::Index<usize> for Syncbuf<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("Index out-of-bounds")
    }
}

impl<T> Drop for Syncbuf<T> {
    fn drop(&mut self) {
        // we don't need to worry about thread-safety here since we have `&mut`
        // SAFETY: The `Syncbuf` is never used again
        unsafe {
            Vec::from_raw_parts(
                self.buf.load(Ordering::Relaxed),
                self.len.load(Ordering::Relaxed),
                self.capacity,
            )
        };
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for Syncbuf<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut i = self.iter();
        write!(f, "Syncbuf {} {{ ", self.capacity)?;
        if let Some(elem) = i.next() {
            write!(f, "{:?}", elem)?;
            for elem in i {
                write!(f, ", {:?}", elem)?;
            }
        }
        write!(f, " }}")
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    extern crate std;
    use alloc::string::{String, ToString};
    #[test]
    fn push() {
        let buf: Syncbuf<usize> = Syncbuf::with_capacity(4);
        let pass = vec![buf.push(400), buf.push(1), buf.push(2), buf.push(9)];
        assert_eq!(pass, vec![Ok(0), Ok(1), Ok(2), Ok(3)]);
        let reject = buf.push(33);
        assert_eq!(reject, Err(33));
    }

    #[test]
    fn refs_not_invalidated() {
        let buf: Syncbuf<String> = Syncbuf::with_capacity(4);
        assert_eq!(buf.push("held".to_string()), Ok(0));
        let held = buf.get(0).unwrap();
        assert_eq!(buf.push("added".to_string()), Ok(1));
        assert_eq!(held, "held");
    }

    #[test]
    fn from_vec() {
        let v = vec!["foo", "bar", "bat"];
        let vec_len = v.len();
        let buf: Syncbuf<_> = v.into();
        assert_eq!(buf.len(), vec_len);
        assert_eq!(buf.get(2), Some(&"bat"));
    }

    #[test]
    #[should_panic(expected = "Index out-of-bounds")]
    fn index() {
        let buf: Syncbuf<_> = vec!["6", "9"].into();
        assert_eq!(buf[0], "6");
        assert_eq!(buf[1], "9");
        assert_eq!(buf[2], "panik");
    }

    #[test]
    fn many_threads() {
        use std::sync::Arc;
        use std::thread::{sleep, JoinHandle};
        use std::time::Duration;

        const THREADS: usize = 20;
        const PUSHES: usize = 200;

        let buf: Syncbuf<i32> = Syncbuf::with_capacity(100000);
        let buf_arc = Arc::new(buf);
        let mut children: Vec<JoinHandle<_>> = Vec::new();
        for i in 0..THREADS as i32 {
            let buf_arc = Arc::clone(&buf_arc);
            children.push(std::thread::spawn(move || {
                for _ in 0..PUSHES {
                    sleep(Duration::from_millis(2));
                    let idx = buf_arc.push(i).unwrap();
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
        assert_eq!(buf_arc.len(), THREADS * PUSHES);
        // assert there are no pushes in progress
        assert_eq!(buf_arc.working_len.load(Ordering::Relaxed), THREADS * PUSHES);

    }
}
