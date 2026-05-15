#![cfg_attr(verus_keep_ghost, verifier::exec_allows_no_decreases_clause)]
use verus_state_machines_macros::tokenized_state_machine;
use verus_builtin::*;
use verus_builtin_macros::*;
use std::sync::Arc;
use std::cmp::Ordering;
use vstd::{
    atomic_ghost::*,
    modes::*,
    prelude::*,
    thread::*,
    pervasive::*, 
    prelude::*, 
    cell::pcell_maybe_uninit::{
        PCell,
        PointsTo
    },
};

verus! {

pub enum NodeData {
    Nil,
    Data(u32)
}

impl NodeData {
    pub fn clone(&self) -> (cloned: Self) 
        ensures
            *self == cloned
    {
        match self {
            NodeData::Nil => NodeData::Nil,
            NodeData::Data(i) => NodeData::Data(*i),
        }
    }

    pub fn get(&self) -> (value: u32) 
        requires
            *self != NodeData::Nil
        ensures
            *self == NodeData::Data(value)
    {
        match self {
            NodeData::Data(i) => *i,
            _ => 0
        }
    }

    pub open spec fn spec_lt(self, other: Self) -> bool {
        match (self, other) {
            (NodeData::Nil, NodeData::Nil) => false,
            (NodeData::Nil, _) => true,
            (_, NodeData::Nil) => false,
            (NodeData::Data(a), NodeData::Data(b)) => a < b,
        }
    }

    pub open spec fn spec_gt(self, other: Self) -> bool {
        match (self, other) {
            (NodeData::Nil, NodeData::Nil) => false,
            (NodeData::Nil, _) => false,
            (_, NodeData::Nil) => true,
            (NodeData::Data(a), NodeData::Data(b)) => a > b,
        }
    }
}

tokenized_state_machine!{
    machine {
        fields {
            #[sharding(map)]
            pub nodes: Map<NodeData, Option<NodeData>>,
            
            #[sharding(variable)]
            pub initialized: bool,
        }

        #[invariant]
        pub fn sorted_inv(&self) -> bool {
            (
                // If the map is initialised with real data
                (self.initialized && self.nodes[NodeData::Nil] != None::<NodeData>) ==> 
                    (
                        // The nil node points to the smallest element in the list:
                        (
                            forall |i: u32| #![auto] 
                                self.nodes[NodeData::Nil] == Some(NodeData::Data(i)) ==>
                                forall |j: u32| #![auto] j < i ==> !self.nodes.dom().contains(NodeData::Data(j))
                        
                        ) &&

                        // The tail node points to the largest element in the list:
                        (
                            forall |i: u32| #![auto]
                                (
                                    self.nodes.dom().contains(NodeData::Data(i)) && 
                                    self.nodes[NodeData::Data(i)] == None::<NodeData>
                                ) ==>
                                (
                                    (forall |j: u32| #![auto] i < j ==> !self.nodes.dom().contains(NodeData::Data(j)))
                                )
                        ) &&

                        // Everything in the list is sorted (smallest to largest).
                        // Nodes either point to something strictly larger, or to None
                        (
                            forall |i: u32| #![auto] 
                                (
                                    self.nodes.dom().contains(NodeData::Data(i)) && 
                                    self.nodes[NodeData::Data(i)] != None::<NodeData>
                                ) ==> (
                                    (exists |j: u32| #![auto] self.nodes[NodeData::Data(i)] == Some(NodeData::Data(j)) && i < j)
                                )
                        ) &&

                        // No two nodes point to the same data:
                        (
                            forall |i: u32, j: u32| #![auto] 
                                (
                                    self.nodes.dom().contains(NodeData::Data(i)) &&
                                    self.nodes.dom().contains(NodeData::Data(j)) &&
                                    self.nodes[NodeData::Data(i)] == self.nodes[NodeData::Data(j)]
                                ) ==>
                                (
                                    i == j
                                )
                        ) &&

                        // // We must assert that for any mapping [a => c], there are no entries in the map
                        // // with key b such that a < b < c. 
                        (
                            forall |i: u32| #![auto] 
                                (
                                    self.nodes.dom().contains(NodeData::Data(i)) && 
                                    self.nodes[NodeData::Data(i)] != None::<NodeData>
                                ) ==> (
                                    exists |j: u32| #![auto] self.nodes[NodeData::Data(i)] == Some(NodeData::Data(j)) && 
                                    forall |k: u32| #![auto] i < k < j ==> !self.nodes.dom().contains(NodeData::Data(k))
                                )
                        )
                    )
            )
        }

        #[invariant]
        pub fn main_inv(&self) -> bool {
            // If the map is uninitialised, then it doesn't contain anything, not even the nil node (and vice versa)
            (!self.initialized <==> self.nodes.is_empty()) &&

            // If the map is initialised, then it must at least have the nil node:
            // This case looks redundant, but I believe it will help the SMT solver.
            (self.initialized <==> self.nodes.dom().contains(NodeData::Nil)) &&

            // If the map contains [NodeData::Nil => None], then it can't contain anything else
            (
                (self.initialized && self.nodes[NodeData::Nil] == None::<NodeData>) <==> 
                (self.nodes =~= Map::empty().insert(NodeData::Nil, None::<NodeData>))
            )
        }

        init!{
            initialize()
            {
                init nodes = Map::empty();
                init initialized = false;
            }
        }

        transition!{
            create_nil()
            {   
                require(!pre.initialized);
                update initialized = true;
                add nodes += [NodeData::Nil => None];
            }
        }

        transition!{
            insert_at_nil_tail(new_tail: u32)
            {   
                remove nodes -= [NodeData::Nil => None];
                add nodes += [NodeData::Nil => Some(NodeData::Data(new_tail))];
                add nodes += [NodeData::Data(new_tail) => None];
            }
        }

        transition!{
            insert_at_cons_tail(current_tail: u32, new_tail: u32)
            {   
                require(new_tail > current_tail);
                remove nodes -= [NodeData::Data(current_tail) => None];
                add nodes += [NodeData::Data(current_tail) => Some(NodeData::Data(new_tail))];
                add nodes += [NodeData::Data(new_tail) => None];
            }
        }

        transition!{
            insert_inbetween_cons_and_cons(lower_car: u32, insert_car: u32, upper_car: u32)
            {   
                require(lower_car < insert_car);
                require(insert_car < upper_car);
                remove nodes -= [NodeData::Data(lower_car) => Some(NodeData::Data(upper_car))];
                add nodes += [NodeData::Data(lower_car) => Some(NodeData::Data(insert_car))];
                add nodes += [NodeData::Data(insert_car) => Some(NodeData::Data(upper_car))];
            }
        }

        transition!{
            insert_inbetween_nil_and_cons(insert_car: u32, upper_car: u32)
            {   
                require(insert_car < upper_car);
                remove nodes -= [NodeData::Nil => Some(NodeData::Data(upper_car))];
                add nodes += [NodeData::Nil => Some(NodeData::Data(insert_car))];
                add nodes += [NodeData::Data(insert_car) => Some(NodeData::Data(upper_car))];
            }
        }

        transition!{
            delete_cons_tail_after_nil(delete_node: u32)
            {   
                remove nodes -= [NodeData::Nil => Some(NodeData::Data(delete_node))];
                remove nodes -= [NodeData::Data(delete_node) => None];
                add nodes += [NodeData::Nil => None];
            }
        }

        transition!{
            delete_cons_tail_node_after_cons(lower_car: u32, delete_node: u32)
            {   
                remove nodes -= [NodeData::Data(lower_car) => Some(NodeData::Data(delete_node))];
                remove nodes -= [NodeData::Data(delete_node) => None];
                add nodes += [NodeData::Data(lower_car) => None];
            }
        }

        transition!{
            delete_inbetween_nil_and_cons(delete_node: u32, upper_car: u32)
            {   
                remove nodes -= [NodeData::Nil => Some(NodeData::Data(delete_node))];
                remove nodes -= [NodeData::Data(delete_node) => Some(NodeData::Data(upper_car))];
                add nodes += [NodeData::Nil => Some(NodeData::Data(upper_car))];
            }
        }

        transition!{
            delete_inbetween_cons_and_cons(lower_car: u32, delete_node: u32, upper_car: u32)
            {   
                remove nodes -= [NodeData::Data(lower_car) => Some(NodeData::Data(delete_node))];
                remove nodes -= [NodeData::Data(delete_node) => Some(NodeData::Data(upper_car))];
                add nodes += [NodeData::Data(lower_car) => Some(NodeData::Data(upper_car))];
            }
        }

        #[inductive(initialize)]
        fn initialize_inductive(post: Self) { 
        }

        #[inductive(create_nil)]
        fn create_nil_inductive(pre: Self, post: Self) { 
        }

        #[inductive(insert_at_nil_tail)]
        fn insert_at_nil_tail_inductive(pre: Self, post: Self, new_tail: u32) { 
        }

        #[inductive(insert_at_cons_tail)]
        fn insert_at_cons_tail_inductive(pre: Self, post: Self, current_tail: u32, new_tail: u32) { 
        }

        #[inductive(insert_inbetween_cons_and_cons)]
        fn insert_inbetween_cons_and_cons_inductive(pre: Self, post: Self, lower_car: u32, insert_car: u32, upper_car: u32) { 
        }

        #[inductive(insert_inbetween_nil_and_cons)]
        fn insert_inbetween_nil_and_cons_inductive(pre: Self, post: Self, insert_car: u32, upper_car: u32) { 
        }

        #[inductive(delete_cons_tail_after_nil)]
        fn delete_cons_tail_after_nil_inductive(pre: Self, post: Self, delete_node: u32) {                     
            assert(post.nodes =~= Map::empty().insert(NodeData::Nil, None::<NodeData>)) by {
                
                assert(forall |i: u32| #![auto] 
                    !post.nodes.dom().contains(NodeData::Data(i)));

                assert forall |node_data: NodeData| post.nodes.dom().contains(node_data) 
                    implies node_data == NodeData::Nil by {
                        match node_data {
                            NodeData::Nil => {}
                            NodeData::Data(i) => {
                                assert(!post.nodes.dom().contains(NodeData::Data(i)))
                            }
                        }
                    }
            
                assert(post.nodes[NodeData::Nil] == None::<NodeData>);
            };
        }

        #[inductive(delete_cons_tail_node_after_cons)]
        fn delete_cons_tail_node_after_cons_inductive(pre: Self, post: Self, lower_car: u32, delete_node: u32) { 
        }

        #[inductive(delete_inbetween_nil_and_cons)]
        fn delete_inbetween_nil_and_cons_inductive(pre: Self, post: Self, delete_node: u32, upper_car: u32) {
            assert(post.initialized <==> post.nodes.dom().contains(NodeData::Nil));
        }

        #[inductive(delete_inbetween_cons_and_cons)]
        fn delete_inbetween_cons_and_cons_inductive(pre: Self, post: Self, lower_car: u32, delete_node: u32, upper_car: u32) {
        }

        // property!{
        //     correct_cons(locked_cons_perm: Tracked<PointsTo<Cons>>, insert_car: u32) {
        //         // require();
                
        //         // have nodes >= [NodeData::Nil => Some(NodeData::Data(first_node_data))];
        //         birds_eye let tokens = pre.nodes;


        //         // assert(
        //         //     tokens.dom().contains(locked_cons_perm.value().map_token)
        //         // );
        //     }
        // }
    }
}

pub struct Nil {
    pub cdr: Option<Arc<LockedCons>>,
    pub map_token: Tracked<machine::nodes>
}

struct_with_invariants!{
    pub struct LockedNil {
        atomic: AtomicBool<_, Option<PointsTo<Nil>>, _>,
        cell: PCell<Nil>,
        instance: Tracked<machine::Instance>,
    }

    spec fn wf(&self) -> bool 
    {
        invariant on atomic with (cell, instance) is (v: bool, g: Option<PointsTo<Nil>>) {
            match g {
                None => v == true,
                Some(points_to) => {
                    v == false &&
                    points_to.is_init() &&
                    points_to.id() == cell.id() &&
                    points_to.value().map_token@.instance_id() == instance@.id() && 
                    points_to.value().map_token@.key() == NodeData::Nil &&
                    (points_to.value().map_token@.value().is_none() <==> points_to.value().cdr.is_none()) && 
                    (points_to.value().map_token@.value().is_some() ==> 
                        (
                            points_to.value().cdr.unwrap().wf() &&
                            points_to.value().cdr.unwrap().view_instance() == instance &&
                            points_to.value().cdr.unwrap().view_car() == points_to.value().map_token@.value().unwrap()
                        )
                    )
                }
            }
        }
    }
}

impl LockedNil {
    fn new() -> (locked_nil: Self)
        ensures 
            locked_nil.wf(),
    {
        let tracked (
            Tracked(instance),
            Tracked(nodes),
            Tracked(initialized),
        ) = machine::Instance::initialize();

        let tracked map_token;
        proof {
            map_token = instance.create_nil(&mut initialized)
        };

        let node = Nil { cdr: None::<Arc<LockedCons>>, map_token: Tracked(map_token) };
        let (cell, Tracked(perm)) = PCell::new(node);
        let atomic = AtomicBool::new(Ghost((cell, Tracked(instance))), false, Tracked(Some(perm)));

        Self { 
            atomic, 
            cell, 
            instance: Tracked(instance)
        }
    }

    fn acquire_lock(&self) -> (points_to: Tracked<PointsTo<Nil>>)
        requires 
            self.wf(),
        ensures 
            points_to.is_init(),
            points_to.id() == self.cell.id(),
            points_to.value().map_token@.instance_id() == self.instance@.id(), 
            points_to.value().map_token@.key() == NodeData::Nil,
            (points_to.value().map_token@.value().is_none() <==> points_to.value().cdr.is_none()), 
            (points_to.value().map_token@.value().is_some() ==> 
                (
                    points_to.value().cdr.unwrap().wf() &&
                    points_to.value().cdr.unwrap().view_instance() == self.instance &&
                    points_to.value().cdr.unwrap().view_car() == points_to.value().map_token@.value().unwrap()
                )
            ),
            self.wf()
    {
        loop
            invariant self.wf(),
        {
            let tracked mut points_to_opt = None;
            let res = atomic_with_ghost!(
                &self.atomic => compare_exchange(false, true);
                ghost points_to_inv => {
                    tracked_swap(&mut points_to_opt, &mut points_to_inv);
                }
            );
            if res.is_ok() {
                return Tracked(points_to_opt.tracked_unwrap());
            }
        }
    }

    fn release_lock(&self, points_to: Tracked<PointsTo<Nil>>)
        requires
            self.wf(),
            points_to.is_init(),
            points_to.id() == self.cell.id(),
            points_to.value().map_token@.instance_id() == self.instance@.id(), 
            points_to.value().map_token@.key() == NodeData::Nil,
            (points_to.value().map_token@.value().is_none() <==> points_to.value().cdr.is_none()), 
            (points_to.value().map_token@.value().is_some() ==> 
                (
                    points_to.value().cdr.unwrap().wf() &&
                    points_to.value().cdr.unwrap().view_instance() == self.instance &&
                    points_to.value().cdr.unwrap().view_car() == points_to.value().map_token@.value().unwrap()
                )
            )
        ensures
            self.wf()
    {
        atomic_with_ghost!(
            &self.atomic => store(false);
            ghost points_to_inv => {
                points_to_inv = Some(points_to.get());
            }
        );
    }

    fn insert(self: Arc<Self>, insert_car: u32)
        requires
            self.wf()
        ensures
            self.wf()
    {
        // Acquire the lock for the nil node, and view the data inside (without taking)
        let mut nil_perm = self.acquire_lock();
        let nil_view = self.cell.borrow(Tracked(nil_perm.borrow_mut()));

        // If the nil cdr is none, then we must insert here - at the tail
        if (nil_view.cdr.is_none()) {

            let mut nil = self.cell.take(Tracked(nil_perm.borrow_mut()));

            let tracked token_tuple;
            let tracked updated_nil_token;
            let tracked cons_token;

            proof {
                token_tuple = self.instance.borrow().insert_at_nil_tail(insert_car, nil.map_token.get());
                updated_nil_token = token_tuple.0.get();
                cons_token = token_tuple.1.get();
            }

            let locked_cons = LockedCons::new(
                insert_car, 
                Tracked(cons_token), 
                None::<Arc<LockedCons>>, 
                self.instance.clone()
            );

            nil.cdr = Some(Arc::new(locked_cons));
            nil.map_token = Tracked(updated_nil_token);
            self.cell.put(Tracked(nil_perm.borrow_mut()), nil);

            self.release_lock(nil_perm);
            return;
        } 
        else {
            // We check if we need to insert inbetween Nil and the first Cons
            let first_locked_cons = nil_view.cdr.as_ref().unwrap().clone();
            let mut first_cons_perm = first_locked_cons.acquire_lock();
            let first_cons_view = first_locked_cons.cell.borrow(Tracked(first_cons_perm.borrow_mut()));

            // If a Cons with this value already exists:
            if (insert_car == first_cons_view.car) {
                // Return early and do nothing - the Cons exists.
                self.release_lock(nil_perm);
                first_locked_cons.release_lock(first_cons_perm);
                return;
            }

            // If the first Cons cdr is larger than the insert cdr:
            if (insert_car < first_cons_view.car) {

                // Then we insert inbetween Nil and first Cons
                let mut nil = self.cell.take(Tracked(nil_perm.borrow_mut()));

                let tracked token_tuple;
                let tracked updated_nil_token;
                let tracked cons_token;

                proof {
                    token_tuple = self.instance.borrow().insert_inbetween_nil_and_cons(insert_car, first_cons_view.car, nil.map_token.get());
                    updated_nil_token = token_tuple.0.get();
                    cons_token = token_tuple.1.get();
                }

                let locked_cons = LockedCons::new(
                    insert_car, 
                    Tracked(cons_token), 
                    Some(first_locked_cons.clone()), 
                    self.instance.clone()
                );

                nil.cdr = Some(Arc::new(locked_cons));
                nil.map_token = Tracked(updated_nil_token);

                self.cell.put(Tracked(nil_perm.borrow_mut()), nil);

                self.release_lock(nil_perm);
                first_locked_cons.release_lock(first_cons_perm);
                return;
            }

            // If we have reached here, we may release the nil lock:
            self.release_lock(nil_perm);

            // Any insert from here onwards will not involve nil - 
            // we may delegate the insert to a chain of LockedCons
            first_locked_cons.insert(first_cons_perm, insert_car);
        }
    }
}

pub struct Cons {
    pub car: u32,
    pub cdr: Option<Arc<LockedCons>>,
    pub map_token: Tracked<machine::nodes>
}

struct_with_invariants!{
    pub struct LockedCons {
        atomic: AtomicBool<_, Option<PointsTo<Cons>>, _>,
        cell: PCell<Cons>,
        instance: Tracked<machine::Instance>,
        view_car: Ghost<NodeData>,
    }

    pub closed spec fn wf(&self) -> bool {
        invariant on atomic with (cell, instance, view_car) is (v: bool, g: Option<PointsTo<Cons>>) {
            match g {
                None => v == true,
                Some(points_to) => {
                    v == false &&
                    points_to.is_init() &&
                    points_to.id() == cell.id() &&
                    NodeData::Data(points_to.value().car) == view_car &&
                    points_to.value().map_token@.instance_id() == instance@.id() &&
                    points_to.value().map_token@.key() == NodeData::Data(points_to.value().car) &&
                    (points_to.value().map_token@.value().is_none() <==> points_to.value().cdr.is_none()) && 
                    (points_to.value().map_token@.value().is_some() ==> 
                        (
                            points_to.value().cdr.unwrap().wf() &&
                            points_to.value().cdr.unwrap().view_instance() == instance &&
                            points_to.value().cdr.unwrap().view_car() > NodeData::Data(points_to.value().car) &&
                            points_to.value().cdr.unwrap().view_car() == points_to.value().map_token@.value().unwrap()
                        )
                    )
                }
            }
        }
    }
}

impl LockedCons {
    fn new(car: u32, map_token: Tracked<machine::nodes>, cdr: Option<Arc<LockedCons>>, instance: Tracked<machine::Instance>) -> (new_cons: Self)
        requires
            map_token@.instance_id() == instance@.id(),
            map_token@.key() == NodeData::Data(car),
            map_token@.value().is_none() <==> cdr.is_none(),
            map_token@.value().is_some() ==> (
                cdr.unwrap().wf() &&
                cdr.unwrap().view_instance() == instance &&
                cdr.unwrap().view_car() > NodeData::Data(car) &&
                cdr.unwrap().view_car() == map_token@.value().unwrap()
            ),
        ensures 
            new_cons.wf(),
            new_cons.instance == instance,
            new_cons.view_car == NodeData::Data(car),
    {   
        let view_car = Ghost(NodeData::Data(car));
        let node = Cons { car, cdr, map_token: map_token };
        let (cell, Tracked(perm)) = PCell::new(node);
        let atomic = AtomicBool::new(Ghost((cell, instance, view_car)), false, Tracked(Some(perm)));
        Self { atomic, cell, instance, view_car }
    }

    fn acquire_lock(&self) -> (points_to: Tracked<PointsTo<Cons>>)
        requires 
            self.wf(),
        ensures 
            points_to.is_init(),
            points_to.id() == self.cell.id(),
            NodeData::Data(points_to.value().car) == self.view_car,
            points_to.value().map_token@.instance_id() == self.instance@.id(),
            points_to.value().map_token@.key() == NodeData::Data(points_to.value().car),
            (points_to.value().map_token@.value().is_none() <==> points_to.value().cdr.is_none()), 
            (points_to.value().map_token@.value().is_some() ==> 
                (
                    points_to.value().cdr.unwrap().wf() &&
                    points_to.value().cdr.unwrap().view_instance() == self.instance &&
                    points_to.value().cdr.unwrap().view_car() > NodeData::Data(points_to.value().car) &&
                    points_to.value().cdr.unwrap().view_car() == points_to.value().map_token@.value().unwrap()
                )
            ),
            self.wf()
    {
        loop
            invariant self.wf(),
        {
            let tracked mut points_to_opt = None;
            let res = atomic_with_ghost!(
                &self.atomic => compare_exchange(false, true);
                ghost points_to_inv => {
                    tracked_swap(&mut points_to_opt, &mut points_to_inv);
                }
            );
            if res.is_ok() {
                return Tracked(points_to_opt.tracked_unwrap());
            }
        }
    }

    fn release_lock(&self, points_to: Tracked<PointsTo<Cons>>)
        requires
            self.wf(),
            points_to.is_init(),
            points_to.id() == self.cell.id(),
            NodeData::Data(points_to.value().car) == self.view_car,
            points_to.value().map_token@.instance_id() == self.instance@.id(),
            points_to.value().map_token@.key() == NodeData::Data(points_to.value().car),
            (points_to.value().map_token@.value().is_none() <==> points_to.value().cdr.is_none()), 
            (points_to.value().map_token@.value().is_some() ==> 
                (
                    points_to.value().cdr.unwrap().wf() &&
                    points_to.value().cdr.unwrap().view_instance() == self.instance &&
                    points_to.value().cdr.unwrap().view_car() > NodeData::Data(points_to.value().car) &&
                    points_to.value().cdr.unwrap().view_car() == points_to.value().map_token@.value().unwrap()
                )
            ),
        ensures
            self.wf()
    {
        atomic_with_ghost!(
            &self.atomic => store(false);
            ghost points_to_inv => {
                points_to_inv = Some(points_to.get());
            }
        );
    }

    pub closed spec fn view_car(&self) -> (view_car: NodeData)
    {
        self.view_car@
    }

    pub closed spec fn view_instance(&self) -> (instance: machine::Instance)
    {
        self.instance@
    }

    fn insert(self: Arc<Self>, mut current_cons_perm: Tracked<PointsTo<Cons>>, insert_car: u32)
        requires
            self.wf(),
            current_cons_perm.is_init(),
            current_cons_perm.id() == self.cell.id(),
            NodeData::Data(current_cons_perm.value().car) == self.view_car,
            current_cons_perm.value().map_token@.instance_id() == self.instance@.id(),
            current_cons_perm.value().map_token@.key() == NodeData::Data(current_cons_perm.value().car),
            (current_cons_perm.value().map_token@.value().is_none() <==> current_cons_perm.value().cdr.is_none()), 
            (current_cons_perm.value().map_token@.value().is_some() ==> 
                (
                    current_cons_perm.value().cdr.unwrap().wf() &&
                    current_cons_perm.value().cdr.unwrap().view_instance() == self.instance &&
                    current_cons_perm.value().cdr.unwrap().view_car() > NodeData::Data(current_cons_perm.value().car) &&
                    current_cons_perm.value().cdr.unwrap().view_car() == current_cons_perm.value().map_token@.value().unwrap()
                )
            ),
            current_cons_perm.value().car < insert_car
        ensures
            self.wf()
    {
        let mut current_locked_cons = self;
        loop 
            invariant
                self.wf(),
                current_locked_cons.wf(),
                current_cons_perm.is_init(),
                current_cons_perm.id() == current_locked_cons.cell.id(),
                NodeData::Data(current_cons_perm.value().car) == current_locked_cons.view_car,
                current_cons_perm.value().map_token@.instance_id() == current_locked_cons.instance@.id(),
                current_cons_perm.value().map_token@.key() == NodeData::Data(current_cons_perm.value().car),
                (current_cons_perm.value().map_token@.value().is_none() <==> current_cons_perm.value().cdr.is_none()), 
                (current_cons_perm.value().map_token@.value().is_some() ==> 
                    (
                        current_cons_perm.value().cdr.unwrap().wf() &&
                        current_cons_perm.value().cdr.unwrap().view_instance() == current_locked_cons.instance &&
                        current_cons_perm.value().cdr.unwrap().view_car() > NodeData::Data(current_cons_perm.value().car) &&
                        current_cons_perm.value().cdr.unwrap().view_car() == current_cons_perm.value().map_token@.value().unwrap()
                    )
                ),
                current_cons_perm.value().car < insert_car
            decreases
                insert_car - current_cons_perm.value().car
        {
            let mut current_cons_view = current_locked_cons.cell.borrow(Tracked(current_cons_perm.borrow_mut()));

            // If there is no next LockedCons, then we must insert at the tail after a Cons
            if (current_cons_view.cdr.is_none()) {

                let mut old_tail_cons = current_locked_cons.cell.take(Tracked(current_cons_perm.borrow_mut()));

                let tracked token_tuple;
                let tracked updated_old_tail_cons_token;
                let tracked new_tail_cons_token;

                proof {
                    token_tuple = current_locked_cons.instance.borrow().insert_at_cons_tail(current_cons_view.car, insert_car, old_tail_cons.map_token.get());
                    updated_old_tail_cons_token = token_tuple.0.get();
                    new_tail_cons_token = token_tuple.1.get();
                }

                let locked_cons = LockedCons::new(
                    insert_car, 
                    Tracked(new_tail_cons_token), 
                    None::<Arc<LockedCons>>, 
                    current_locked_cons.instance.clone()
                );

                old_tail_cons.cdr = Some(Arc::new(locked_cons));
                old_tail_cons.map_token = Tracked(updated_old_tail_cons_token);

                current_locked_cons.cell.put(Tracked(current_cons_perm.borrow_mut()), old_tail_cons);
                current_locked_cons.release_lock(current_cons_perm);

                return;
            } 
            // Otherwise, there is another LockedCons
            else {
                // Acquire the permissions to access the Cons:
                let next_locked_cons = current_cons_view.cdr.as_ref().unwrap().clone();
                let mut next_cons_perm = next_locked_cons.acquire_lock();
                let next_cons_view = next_locked_cons.cell.borrow(Tracked(next_cons_perm.borrow_mut()));

                // If a Cons with this value already exists:
                if (insert_car == next_cons_view.car) {
                    // Return early and do nothing - the Cons exists.
                    current_locked_cons.release_lock(current_cons_perm);
                    next_locked_cons.release_lock(next_cons_perm);
                    return;
                }

                // If the next Cons cdr is larger than the insert cdr:
                if (insert_car < next_cons_view.car) {

                    // Then we insert inbetween Cons and Cons
                    let mut current_cons = current_locked_cons.cell.take(Tracked(current_cons_perm.borrow_mut()));

                    let tracked token_tuple;
                    let tracked updated_cons_token;
                    let tracked new_cons_token;

                    proof {
                        token_tuple = current_locked_cons.instance.borrow().insert_inbetween_cons_and_cons(current_cons_view.car, insert_car, next_cons_view.car, current_cons.map_token.get());
                        updated_cons_token = token_tuple.0.get();
                        new_cons_token = token_tuple.1.get();
                    }

                    let locked_cons = LockedCons::new(
                        insert_car, 
                        Tracked(new_cons_token), 
                        Some(next_locked_cons.clone()), 
                        current_locked_cons.instance.clone()
                    );

                    current_cons.cdr = Some(Arc::new(locked_cons));
                    current_cons.map_token = Tracked(updated_cons_token);

                    current_locked_cons.cell.put(Tracked(current_cons_perm.borrow_mut()), current_cons);

                    current_locked_cons.release_lock(current_cons_perm);
                    next_locked_cons.release_lock(next_cons_perm);
                    return;
                }

                // Otherwise, we give up the previous lock, and loop again
                current_locked_cons.release_lock(current_cons_perm);

                current_locked_cons = next_locked_cons;
                current_cons_perm = next_cons_perm;
            }
        }
    }
}

pub struct LinkedList {
    pub locked_nil: Arc<LockedNil>,
}

impl LinkedList {
    pub closed spec fn wf(&self) -> bool
    {
        self.locked_nil.wf()
    }

    pub fn new() -> (linked_list: Self)
        ensures
            linked_list.wf()
    {
        Self { locked_nil: Arc::new(LockedNil::new()) }
    }

    pub fn insert(self, data: u32) 
        requires
            self.wf()
        ensures
            self.wf()
    {
        self.locked_nil.insert(data);
    }
}

fn main() {
    let linked_list = Arc::new(LinkedList::new());
}
}