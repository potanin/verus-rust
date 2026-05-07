#![cfg_attr(verus_keep_ghost, verifier::exec_allows_no_decreases_clause)]
use verus_state_machines_macros::tokenized_state_machine;
use verus_builtin::*;
use verus_builtin_macros::*;
use std::sync::Arc;
use std::cmp::Ordering;
use vstd::atomic_ghost::*;
use vstd::modes::*;
use vstd::prelude::*;
use vstd::thread::*;
use vstd::{pervasive::*, prelude::*, *};
use vstd::cell::pcell;
use vstd::set::*;

verus! {

tokenized_state_machine!{
    machine {
        fields {
            #[sharding(set)]
            pub bools: Set<bool>,
        }

        #[invariant]
        pub fn main_inv(&self) -> bool {
            (self.bools.contains(true) <==> !self.bools.contains(false)) &&
            (self.bools.contains(true) || self.bools.contains(false))
        }

        property!{
            have_true() {
                have bools >= set { true };
                birds_eye let s = pre.bools;

                assert(s.contains(true) <==> !s.contains(false));
                assert(!s.contains(false));
            }
        }
    }
}

fn main() { 
}
}