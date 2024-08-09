use std::{
    fmt,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub struct RwSlot<T>(Arc<RwLock<Option<T>>>);

impl<T> Clone for RwSlot<T> {
    fn clone(&self) -> Self {
        self.shallow_clone()
    }
}

impl<T: Clone> RwSlot<T> {
    pub fn deep_clone(&self) -> Self {
        let inner = self.lock_read();
        match inner.as_ref() {
            Some(value) => RwSlot::new(value.clone()),
            None => RwSlot::none(),
        }
    }
}

impl<T> fmt::Display for RwSlot<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            write!(f, "Empty or closed slot")
        } else {
            write!(f, "{}", self.read_inner().deref())
        }
    }
}

impl<T> RwSlot<T> {
    pub fn new(x: T) -> Self {
        RwSlot(Arc::new(RwLock::new(Some(x))))
    }

    pub fn none() -> Self {
        RwSlot(Arc::new(RwLock::new(None)))
    }

    pub fn is_none(&self) -> bool {
        self.lock_read().is_none()
    }

    pub fn lock_read(&self) -> RwLockReadGuard<'_, Option<T>> {
        self.0.read()
    }

    pub fn lock_read_recursive(&self) -> RwLockReadGuard<'_, Option<T>> {
        self.0.read_recursive()
    }

    pub fn lock_write(&self) -> RwLockWriteGuard<'_, Option<T>> {
        self.0.write()
    }

    pub fn read_inner(&self) -> ReadInner<'_, T> {
        ReadInner(self.0.read())
    }

    pub fn write_inner(&self) -> WriteInner<'_, T> {
        WriteInner(self.0.write())
    }

    pub fn insert(&self, data: T) -> Option<T> {
        std::mem::replace(&mut self.lock_write(), Some(data))
    }

    pub fn extract(&self) -> Option<T> {
        self.lock_write().take()
    }

    pub fn shallow_clone(&self) -> Self {
        RwSlot(Arc::clone(&self.0))
    }

    pub fn drop(&self) {
        let _ = self.extract();
    }

    pub fn swap(&self, other: &Self) {
        let mut self_lock = self.lock_write();
        let mut other_lock = other.lock_write();
        std::mem::swap(self_lock.deref_mut(), other_lock.deref_mut());
    }
}

pub struct ReadInner<'a, T>(pub RwLockReadGuard<'a, Option<T>>);

impl<T> Deref for ReadInner<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("accessing an empty slot")
    }
}

pub struct WriteInner<'a, T>(pub RwLockWriteGuard<'a, Option<T>>);

impl<T> Deref for WriteInner<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("accessing an empty slot")
    }
}

impl<T> DerefMut for WriteInner<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().expect("accessing an empty slot")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_slot_with_value() {
        let slot = RwSlot::new(10);
        assert!(!slot.is_none());
        assert_eq!(*slot.read_inner(), 10);
    }

    #[test]
    fn create_empty_slot() {
        let slot: RwSlot<i32> = RwSlot::none();
        assert!(slot.is_none());
    }

    #[test]
    fn insert_and_extract_value() {
        let slot = RwSlot::new(10);
        assert_eq!(slot.extract(), Some(10));
        assert!(slot.is_none());
    }

    #[test]
    fn insert_value_into_empty_slot() {
        let slot: RwSlot<i32> = RwSlot::none();
        assert!(slot.is_none());
        assert_eq!(slot.insert(20), None);
        assert_eq!(*slot.read_inner(), 20);
    }

    #[test]
    fn swap_slots() {
        let slot1 = RwSlot::new(10);
        let slot2 = RwSlot::new(20);
        slot1.swap(&slot2);
        assert_eq!(*slot1.read_inner(), 20);
        assert_eq!(*slot2.read_inner(), 10);
    }

    #[test]
    fn modify_inner_value() {
        let slot = RwSlot::new(10);
        {
            let mut inner = slot.write_inner();
            *inner = 30;
        }
        assert_eq!(*slot.read_inner(), 30);
    }

    #[test]
    fn display_slot() {
        let slot = RwSlot::new(10);
        assert_eq!(format!("{}", slot), "10");
        slot.extract();
        assert_eq!(format!("{}", slot), "Empty or closed slot");
    }

    #[test]
    fn test_shallow_clone() {
        let original = RwSlot::new(vec![1, 2, 3]);
        let cloned = original.shallow_clone();

        // Verify that the cloned value is equal to the original
        assert_eq!(*original.read_inner(), *cloned.read_inner());

        // Modify the original to ensure the clone shares the same data
        original.write_inner().push(4);

        // Verify that the modification affects both original and clone
        assert_eq!(*original.read_inner(), vec![1, 2, 3, 4]);
        assert_eq!(*cloned.read_inner(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_deep_clone() {
        let original = RwSlot::new(vec![1, 2, 3]);
        let cloned = original.deep_clone();

        // Verify that the cloned value is equal to the original
        assert_eq!(*original.read_inner(), *cloned.read_inner());

        // Modify the original to ensure the clone is independent
        original.write_inner().push(4);

        // Verify that the modification doesn't affect the clone
        assert_eq!(*original.read_inner(), vec![1, 2, 3, 4]);
        assert_eq!(*cloned.read_inner(), vec![1, 2, 3]);
    }

    #[test]
    fn test_default_clone() {
        let original = RwSlot::new(vec![1, 2, 3]);
        let cloned = original.clone();

        // Verify that the cloned value is equal to the original
        assert_eq!(*original.read_inner(), *cloned.read_inner());

        // Modify the original to ensure the clone shares the same data
        original.write_inner().push(4);

        // Verify that the modification affects both original and clone
        assert_eq!(*original.read_inner(), vec![1, 2, 3, 4]);
        assert_eq!(*cloned.read_inner(), vec![1, 2, 3, 4]);
    }
}
