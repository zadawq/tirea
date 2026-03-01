use serde::{Deserialize, Serialize};
use tirea_state_derive::State;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, State)]
struct Bad {
    #[tirea(lattice, flatten)]
    field: GCounter,
}

struct GCounter;

fn main() {}
