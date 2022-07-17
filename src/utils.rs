pub mod prelude {
    pub use crate::error::{Error,Result};
    pub use super::iter_tools::GenerateIter;
}

pub mod iter_tools {
    pub trait GenerateIter<F: FnMut() -> Option<O>,O> {
        fn generate_iter(self) -> GenIter<F,O>;
    }

    impl<F: FnMut() -> Option<O>,O> GenerateIter<F,O> for F {
        fn generate_iter(self) -> GenIter<F,O> {
            GenIter(self)
        }
    }

    pub struct GenIter<F: FnMut() -> Option<O>,O>(F);

    impl<F: FnMut() -> Option<O>,O> Iterator for GenIter<F,O> {
        type Item = O;
        fn next(&mut self) -> Option<Self::Item> {
            (self.0)()
        }
    }
}

pub trait Transpose {
    fn transpose(self) -> Self;
}

impl<T> Transpose for Vec<Vec<T>> where T: Sized {
    fn transpose(mut self) -> Self {
        let width = self.len();
        if width == 0 { vec![] } else {
            let height = self.first().unwrap().len();
            let mut out = Vec::with_capacity(height);
            for mut column in self.drain(..) {
                for (y,cell) in column.drain(..).enumerate() {
                    if out.len() == y {
                        out.push(Vec::with_capacity(width));
                    }
                    out[y].push(cell);
                }
            }
            out
        }
    }
}