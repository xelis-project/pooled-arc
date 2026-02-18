## Pooled Arc

The `pooled-arc` library provides a thread-safe, memory-efficient way to manage shared objects in Rust using a combination of `Arc` and `Weak` references.

It is designed to de-duplicate objects by storing them in a global pool. When a new object is created, the library checks if an identical object already exists in the pool and returns a shared reference to it instead of creating a new one. This can significantly reduce memory usage and improve performance when dealing with large numbers of identical objects.

## Getting Started

To use the `pooled-arc` library, add it to your `Cargo.toml`:

```toml
[dependencies]
pooled-arc = "0.1"
```

## Usage

Here is a simple example of how to use the `pooled-arc` library:

```rust
use pooled_arc::*;

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct Foo {
    id: u64,
    name: String,
}

// Test type with its own static pool
impl_internable!(Foo);

fn main() {
    let val = Foo { id: 1, name: "Hello World!".to_string() };
    let a = PooledArc::new(val.clone());

    // Create a second instance with the same value, should return the same Arc
    let b = PooledArc::new(val);

    // The inner values are the same
    assert_eq!(a.as_ptr(), b.as_ptr());
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.