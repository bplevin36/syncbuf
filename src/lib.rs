use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::sync::atomic::{AtomicUsize, AtomicPtr, Ordering, spin_loop_hint};

/// A fixed-size, thread-safe buffer that allows adding new
/// elements without invalidating existing references to previous elements
pub struct Syncbuf<T> {
    capacity: usize,
    len: AtomicUsize,
    working_len: AtomicUsize,
    buf: AtomicPtr<T>,
}

impl<T> Syncbuf<T> {
    pub fn with_capacity(capacity: usize) -> Syncbuf<T> {
        let layout = Layout::array::<T>(capacity).unwrap();
        // Safety: we ensure allocation succeeded
        let allocation = unsafe { alloc(layout) };
        if allocation.is_null() {
            handle_alloc_error(layout);
        }
        let buf: AtomicPtr<T> = AtomicPtr::new(allocation.cast::<T>());
        Syncbuf {
            capacity,
            buf,
            len: AtomicUsize::new(0),
            working_len: AtomicUsize::new(0),
        }
    }

    /// Pushes an element at the end of the buffer, returning its index
    ///
    /// Returns `Ok(index)` if the push was successful.
    ///
    /// Returns `Err(elem)` if the buffer was already full
    pub fn push(&self, elem: T) -> Result<usize, T> {
        // We only ever write to the buffer:
        //  - After incrementing working_len
        //  - At the index working_len - 1
        let idx = self.working_len.fetch_add(1, Ordering::SeqCst);
        if idx >= self.capacity {
            return Err(elem);
        }

        // SAFETY: we have checked that `idx` is within the buffer and
        // we will never write to the same `idx` twice
        unsafe {
            let location = self.buf.load(Ordering::Relaxed).add(idx) ;
            location.write(elem);
        }

        // It is possible the element we just wrote was more than 1 past `len`
        // if this is the case, we have to wait until the concurrent push finishes
        while self.len.compare_and_swap(idx, idx + 1, Ordering::SeqCst) != idx {
            spin_loop_hint();
        }
        Ok(idx)
    }

    // Gets a reference to an element.
    pub fn get(&self, idx: usize) -> Option<&T> {
        let len = self.len.load(Ordering::SeqCst);
        if idx >= len {
            None
        } else {
            // SAFETY: `idx` < `len` < `capacity` so it is guaranteed to be both
            // non-null and never written again
            let elem_ptr = unsafe { self.buf.load(Ordering::Relaxed).add(idx) };
            let elem_ref = unsafe { elem_ptr.as_ref() };
            elem_ref
        }
    }
}

impl<T> Drop for Syncbuf<T> {
    fn drop(&mut self) {
        // we don't need to be very thread-safe here since we have `&mut`
        let beginning = self.buf.load(Ordering::Relaxed);
        for i in 0..self.len.load(Ordering::SeqCst) {
            // SAFETY: all shared references to items in the buffer have their
            // lifetimes tied to the buffer, so none exist by now and it is safe
            // to drop all contents
            unsafe {
                std::ptr::drop_in_place(beginning.add(i));
            }
        }
        // SAFETY: after we deallocate, the `Syncbuf` is never used again
        unsafe {
            dealloc(beginning as *mut u8, Layout::array::<T>(self.capacity).unwrap());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(buf.push("held".to_owned()), Ok(0));
        let held = buf.get(0).unwrap();
        assert_eq!(buf.push("added".to_owned()), Ok(1));
        assert_eq!(held, "held");
    }
}
