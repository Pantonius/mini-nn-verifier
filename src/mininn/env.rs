use std::collections::BTreeMap;

/// Generic mapping of variable names to *something*.
/// For example: Env<f64> maps variable names to concrete floating-point values
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

    pub fn update(&mut self, key: String, new_value: T) -> bool {
        let opt_val = self.inner.get_mut(&key);

        if opt_val.is_none() {
            return false;
        }

        *opt_val.unwrap() = new_value;

        return true;
    }
}
