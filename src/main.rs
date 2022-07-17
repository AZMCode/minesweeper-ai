#![feature(type_alias_impl_trait)]
#![feature(generic_associated_types)]
#![feature(array_try_map)]

mod error;
mod utils;
mod game;
mod ai;
mod app;

fn main() -> ! { eframe::run_native("Minesweeper", eframe::NativeOptions::default(), Box::new(|_| Box::new(app::App::new().unwrap()))) }
