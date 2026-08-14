use std::collections::BTreeMap;

/// Maps variable names to their computed tensor values.
pub struct Env<T> {
    inner: BTreeMap<String, T>,
}
impl<T> Env<T> {
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }
    pub fn get(&self, key: &str) -> Option<&T> {
        self.inner.get(key)
    }
    pub fn insert(&mut self, key: String, value: T) {
        self.inner.insert(key, value);
    }
}
