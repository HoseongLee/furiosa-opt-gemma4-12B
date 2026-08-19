
#![expect(clippy::type_complexity)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::prelude::*;

pub mod api;
pub mod axes;
pub mod host;
pub mod ops;
pub mod ops_audio;
pub mod ops_vision;

pub(crate) mod device;

pub type Chip = m![1];

pub const LAYERS: usize = 48;
pub const EMBED_SCALE: f32 = 61.967_734;
pub const LOGIT_SOFTCAP: f32 = 30.0;
pub const EPS: f32 = 1e-6;
