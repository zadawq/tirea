use serde::{Deserialize, Serialize};
use tirea_state_derive::State;

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, State)]
struct Bad {
    #[tirea(lattice)]
    field: BTreeMap<String, GCounter>,
}

struct GCounter;

fn main() {}
