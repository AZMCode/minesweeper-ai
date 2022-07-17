type Result<T = (), E = AIError> = std::result::Result<T,E>;

#[derive(Clone,Copy)]
pub struct MineToClearRatio(pub usize,pub usize);

impl MineToClearRatio {
    fn to_pair(self) -> (usize,usize) {
        <(usize,usize) as From<_>>::from(self)
    }
    fn from_pair(pair: (usize,usize)) -> Self {
        Self::from(pair)
    }
    fn ratio_error(self,other: Self) -> f64 {
        let (lhs_num,lhs_den) = (self.0 , self.1 );
        let lhs_tot = lhs_num + lhs_den;
        let (rhs_num,rhs_den) = (other.0, other.1);
        let rhs_tot = rhs_num + rhs_den;
        ((lhs_num * rhs_tot).abs_diff(rhs_num * lhs_tot) as f64) / ((lhs_tot * rhs_tot) as f64)
    }
}

impl std::fmt::Debug for MineToClearRatio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("({}:{})",self.0,self.1))
    }
}

impl From<(usize,usize)> for MineToClearRatio {
    fn from((a,b): (usize,usize)) -> Self {
        MineToClearRatio(a,b)
    }
}

impl From<MineToClearRatio> for (usize,usize) {
    fn from(MineToClearRatio(a,b): MineToClearRatio) -> Self {
        (a,b)
    }
}

impl PartialEq for MineToClearRatio {
    fn eq(&self, other: &Self) -> bool {
        match (self.to_pair(),other.to_pair()) {
            ((0,0),_) | (_,(0,0)) => false,
            ((a,b),(c,d)) => (a*d).eq(&(c*b))
        }
    }
}

impl PartialOrd for MineToClearRatio {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self.to_pair(),other.to_pair()) {
            ((0,0),_) | (_,(0,0))   => None,
            ((a,b),(c,d)) => Some((a*d).partial_cmp(&(c*b)).unwrap())
        }
    }
}

impl From<Cell> for MineToClearRatio {
    fn from(input: Cell) -> Self {
        match input {
            Cell::Open(crate::game::MCellContents::Mine) => MineToClearRatio(1, 0),
            Cell::Open(crate::game::MCellContents::Number(_)) => MineToClearRatio(0,1),
            Cell::Covered(inner_ratio) => inner_ratio
        }
    }
}

impl std::ops::Add for MineToClearRatio {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        MineToClearRatio(self.0 + rhs.0,self.1 + rhs.1)
    }
}

impl std::iter::Sum for MineToClearRatio {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(MineToClearRatio(0,0), |acc,elm| acc + elm)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AIError {
    #[error("No such board with the name '{0}'")]
    NoSuchBoard(String)
}

trait CellInfo {
    fn classify(&self) -> CellClassification;
    fn is_defined(&self) -> bool {
        use CellClassification::*;
        match self.classify() {
            DefinedLeft | DefinedRight => true,
            _ => false
        }
    }
    fn is_partial(&self) -> bool {
        use CellClassification::*;
        match self.classify() {
            Partial => true,
            _ => false
        }
    }
}
enum CellClassification {
    DefinedLeft,
    DefinedRight,
    FullyUndefined,
    Partial
}

impl CellInfo for MineToClearRatio {
    fn classify(&self) -> CellClassification {
        use CellClassification::*;
        match (self.0,self.1) {
            (other,0) if other != 0 => DefinedLeft,
            (0,other) if other != 0 => DefinedRight,
            (0,0) => FullyUndefined,
            _ => Partial
        }
    }
}

#[derive(Debug,Clone)]
pub enum Cell {
    Covered(MineToClearRatio),
    Open(crate::game::MCellContents)
}

impl From<crate::game::MCell> for Cell {
    fn from(input: crate::game::MCell) -> Self {
        use crate::game::MCell::*;
        match input {
            Open(contents) => Cell::Open(contents),
            Covered => Cell::Covered(MineToClearRatio(0, 0))
        }
    }
}

impl CellInfo for crate::game::MCellContents {
    fn classify(&self) -> CellClassification {
        use crate::game::MCellContents::*;
        match self {
            Mine => CellClassification::DefinedLeft,
            Number(_) => CellClassification::DefinedRight
        }
    }
}

impl CellInfo for Cell {
    fn classify(&self) -> CellClassification {
        use Cell::*;
        match self {
            Covered(ratio) => ratio.classify(),
            Open(contents) => contents.classify()
        }
    }
}

impl From<Cell> for crate::game::MCell {
    fn from(input: Cell) -> Self {
        use Cell::*;
        use crate::game::MCell;
        match input {
            Covered(_) => MCell::Covered,
            Open(contents) => MCell::Open(contents)
        }
    }
}

macro_rules! any_board {
    ($($id: ident),*) => {
            pub enum AnyBoard {
                $($id(Board<strategy::$id>)),*
            }

        pub const STRATEGY_NAMES: &'static [&'static str] = &[$(stringify!($id)),*];

        impl AnyBoard {
            pub fn new<I,C>(board_kind: &str, width: usize, height: usize, mines: usize, cells_iter: I) -> Result<Self> where
                I: IntoIterator<Item = C>,
                C: Into<crate::game::MCell>
            {
                match board_kind {
                    $(stringify!($id) => Ok(AnyBoard::$id(<Board<strategy::$id>>::new::<I,C>(width,height,mines,cells_iter)?))),*,
                    _ => Err(AIError::NoSuchBoard(board_kind.to_string()))
                
                }
            }

            pub fn refine_probabilities(&mut self) {
                match self {
                    $(AnyBoard::$id(v) => v.refine_probabilities()),*
                }
            }

            pub fn naive_probabilities(&self) -> Vec<MineToClearRatio> {
                match self {
                    $(AnyBoard::$id(v) => v.naive_probabilities()),*
                }
            }

            pub fn update_board<I,C>(&mut self, cells_iter: I) where
                I: IntoIterator<Item = C>,
                C: Into<crate::game::MCell>
            {
                match self {
                    $(AnyBoard::$id(v) => v.update_board::<I,C>(cells_iter)),*
                }
            }

            pub fn best_choices(&self) -> Option<(MineToClearRatio,Vec<[usize;2]>)> {
                match self {
                    $(AnyBoard::$id(v) => v.best_choices()),*
                }
            }

            pub fn pick_move(&self) -> MoveType {
                match self {
                    $(AnyBoard::$id(v) => v.pick_move()),*
                }
            }

            pub fn iter_slice<X,Y>(&self, range_x: X, range_y: Y) -> BoardSliceIter<'_> where
                X: std::ops::RangeBounds<usize>,
                Y: std::ops::RangeBounds<usize>
            {
                match self {
                    $(AnyBoard::$id(v) => v.iter_slice::<X,Y>(range_x,range_y)),*
                }
            }
        }
    }
}

any_board!(BruteForce, Border, Random, Localized);



pub struct Board<S: strategy::Strategy> {
    width: usize,
    height: usize,
    mines: usize,
    cells: Vec<Cell>,
    strategy: S
}

#[derive(Clone)]
pub enum MoveType {
    Certain(Vec<[usize;2]>),
    BestChanceRandom(Vec<[usize;2]>),
    GivenUp
}

impl<S> Board<S> where
    S: strategy::Strategy
{
    fn refine_probabilities(&mut self) {
        S::refine_probabilities(self)
    }
    pub fn new<I,C>(width: usize, height: usize, mines: usize, cells_iter: I) -> Result<Self> where
        I: IntoIterator<Item = C>,
        C: Into<crate::game::MCell>
    {

        let mut out = Board {
            width, height, mines, cells: vec![], strategy: S::setup(width,height,mines)
        };
        out.update_board(cells_iter);
        Ok(out)
    }
    pub fn naive_probabilities(&self) -> Vec<MineToClearRatio> {
        use crate::game::MCellContents;
        self.cells.iter().map(|c| match c {
            Cell::Covered(_) => MineToClearRatio(0, 0),
            Cell::Open(MCellContents::Mine) => MineToClearRatio(1,0),
            Cell::Open(MCellContents::Number(_)) => MineToClearRatio(0,1)
        }).collect::<Vec<_>>()
    }
    pub fn update_board<I,C>(&mut self, cells_iter: I) where
        I: IntoIterator<Item = C>,
        C: Into<crate::game::MCell>
    {
        self.cells = cells_iter.into_iter().map(|c| Cell::from(c.into())).collect::<Vec<_>>();
        self.refine_probabilities();
        if self.cells.len() != self.width * self.height  {
            panic!("Amount of cells in struct not matching expected value: Actual({}) != Expected({})",
                self.cells.len(),
                self.width * self.height
            )
        }
    }
    pub fn best_choices(&self) -> Option<(MineToClearRatio,Vec<[usize;2]>)> {
        S::best_choices(self)
    }
    pub fn pick_move(&self) -> MoveType {
        match self.best_choices().and_then(|(ratio,coords)| Some((ratio.classify(),coords))) {
            Some((CellClassification::DefinedRight  , coords))  => MoveType::Certain(coords),
            Some((CellClassification::Partial       , coords))  => MoveType::BestChanceRandom(coords),
            None                                                => MoveType::GivenUp,
            Some((CellClassification::DefinedLeft   , _     ))  => MoveType::GivenUp,
            Some((CellClassification::FullyUndefined, _     ))  => panic!("Unexpected undefined cell")
        }
    }
    pub fn iter_slice<X,Y>(&self, range_x: X, range_y: Y) -> BoardSliceIter<'_> where
        X: std::ops::RangeBounds<usize>,
        Y: std::ops::RangeBounds<usize>
    {
        enum Dimension { X, Y }
        enum BoundKind { Start, End}
        use Dimension::*;
        use BoundKind::*;
        use std::ops::Bound::*;
        let bounds = [
            [
                (X,Start,range_x.start_bound()),
                (X,End  ,range_x.end_bound()  )
            ],
            [
                (Y,Start,range_y.start_bound()),
                (Y,End  ,range_y.end_bound()  )
            ]
        ];
        let clipped_bounds = bounds.map(|inner_arr| inner_arr.map(|(dim,kind,bound)| {
            let dim_size =  match dim {
                X => self.width,
                Y => self.height
            };
            match (bound,kind) {
                (Included(v), Start) | (Excluded(v), End  ) => *v,
                (Included(v), End  ) | (Excluded(v), Start) => *v + 1,
                (Unbounded, Start) => 0,
                (Unbounded, End)   => dim_size
            }
        }));
        BoardSliceIter {
            width: self.width,
            height: self.height,
            cells_ref: &self.cells,
            index: Some([clipped_bounds[0][0],clipped_bounds[1][0]]),
            bounds: clipped_bounds
        }
    }
}

pub struct BoardSliceIter<'g> {
    width: usize,
    height: usize,
    cells_ref: &'g [Cell],
    index: Option<[usize;2]>,
    bounds: [[usize;2];2]
}

impl<'g> Iterator for BoardSliceIter<'g> {
    type Item = ([usize;2],&'g Cell);
    fn next(&mut self) -> Option<Self::Item> {
        let width = self.width;
        let mut nullify_index = false;
        let out = match self.index {
            Some([ref mut x, ref mut y]) => {
                let out_ref = &self.cells_ref[width * *y + *x];
                let out = Some(([*x,*y],out_ref));
                *x += 1;
                if *x >= self.bounds[0][1] {
                    *x = self.bounds[0][0];
                    *y += 1;
                }
                if *y >= self.bounds[1][1] {
                    nullify_index = true
                }
                out
            },
            None => None
        };
        if nullify_index {
            self.index = None;
        }
        out
    }
}

pub mod strategy {
    use rand::SeedableRng;

    use crate::utils::Transpose;
    use super::{Board,Cell,MineToClearRatio,CellClassification,CellInfo};

    pub trait Strategy: Sized {
        fn setup(width: usize, height: usize, mines: usize) -> Self;
        fn refine_probabilities(board: &mut Board<Self>);
        fn best_choices(board: &Board<Self>) -> Option<(MineToClearRatio,Vec<[usize;2]>)> {
            use std::cmp::Ordering;
            board.cells
                .iter()
                .enumerate()
                .fold(vec![], |acc: Vec<(usize,MineToClearRatio)>,(index,elm)|
                    match elm {
                        Cell::Open(_) => acc,
                        Cell::Covered(ratio) => match acc.as_slice() {
                            [] => vec![(index,*ratio)],
                            [(first_index,first_ratio), rest @ .. ] => match ratio.partial_cmp(first_ratio) {
                                Some(Ordering::Less)    => vec![(index,*ratio)],
                                Some(Ordering::Equal)   => vec![
                                    vec![(*first_index,*first_ratio)],
                                    rest.to_vec(),
                                    vec![(index,*ratio)]
                                ].into_iter().flatten().collect::<Vec<_>>(),
                                Some(Ordering::Greater) => acc,
                                None => panic!("Unexpected undefined ratio")
                            }
                        }
                    }
                ).into_iter()
                .fold(None,|acc,(offset,ratio)|
                    match acc {
                        None => Some((ratio,vec![crate::game::Game::cell_coord(board.width,offset)])),
                        Some((first_ratio,mut list)) => {
                            list.push(crate::game::Game::cell_coord(board.width,offset));
                            Some((first_ratio,list))
                        }
                    }
                )
        }
    }
    pub struct BruteForce;

    impl Strategy for BruteForce {
        fn setup(_:usize, _: usize, _: usize) -> Self {
            BruteForce
        }
        fn refine_probabilities(board: &mut super::Board<Self>) {
            if board.cells.iter().all(|c| match c { Cell::Covered(_) => true, _ => false }) {
                for cell in board.cells.iter_mut() {
                    *cell = Cell::Covered(MineToClearRatio(1,1))
                }
                return;
            }

            let mut stack = vec![board.cells.clone()];
            let mut poss = vec![];

            while let Some(board_cells) = stack.pop() {
                let first_undefined_offset = board_cells.iter()
                    .enumerate()
                    .find_map(|(offset,cell)|
                        if !cell.is_defined() {
                            Some(offset)
                        } else { None }
                    );
                if let Some(offset) = first_undefined_offset {
                    let mut clear_board = board_cells.clone();
                    clear_board[offset] = Cell::Covered(MineToClearRatio(0, 1));
                    let mut mine_board = board_cells.clone();
                    mine_board[offset]  = Cell::Covered(MineToClearRatio(1, 0));

                    stack.extend(vec![mine_board,clear_board].into_iter().filter(|new_board| {
                        let mut is_valid_board = true;
                        let mut total_mine_count = 0;
                        for (coord,cell) in new_board.iter().enumerate().map(|(o,c)| (crate::game::Game::cell_coord(board.width, o),c)) {
                            match cell {
                                Cell::Open(crate::game::MCellContents::Number(n)) => {
                                    let mut neighbor_count = 0;
                                    for neighbor in crate::game::Game::get_neighbors([board.width,board.height], coord) {
                                        if let CellClassification::DefinedLeft = new_board[crate::game::Game::cell_offset(board.width,neighbor)].classify() {
                                            neighbor_count += 1;
                                        }
                                    }
                                    if &neighbor_count > n {
                                        is_valid_board = false;
                                        break;
                                    }
                                },
                                c if matches!(c.classify(),CellClassification::DefinedLeft) => if total_mine_count >= board.mines {
                                        is_valid_board = false;
                                        break;
                                } else {
                                    total_mine_count += 1;
                                },
                                _ => ()
                            }
                        }
                        is_valid_board
                    }).collect::<Vec<_>>())
                } else {
                    let mut valid_board = true;
                    let mut total_mine_count = 0;
                    for (coord,cell) in board_cells.iter().enumerate().map(|(o,c)| (crate::game::Game::cell_coord(board.width, o),c)) {
                        match cell {
                            Cell::Open(crate::game::MCellContents::Number(n)) => {
                                let mut neighbor_count = 0;
                                for neighbor in crate::game::Game::get_neighbors([board.width,board.height], coord) {
                                    if let CellClassification::DefinedLeft = board_cells[crate::game::Game::cell_offset(board.width,neighbor)].classify() {
                                        neighbor_count += 1;
                                    }
                                }
                                if &neighbor_count != n {
                                    valid_board = false;
                                    break;
                                }
                            },
                            c if matches!(c.classify(),CellClassification::DefinedLeft) => total_mine_count += 1,
                            _ => ()
                        }
                    }
                    if total_mine_count != board.mines {
                        valid_board = false;
                    }
                    if valid_board {
                        poss.push(board_cells.into_iter().map(|c| MineToClearRatio::from(c)).collect::<Vec<_>>());
                    }
                }
            }
            let size = board.height * board.width;
            let mut transposed_possibilities = vec![vec![MineToClearRatio(0, 0);poss.len()];poss[0].len()];
            for offset in 0..size {
                for poss_index in 0..poss.len() {
                    transposed_possibilities[offset][poss_index] = poss[poss_index][offset];
                }
            }
            let refined_possibilities = transposed_possibilities
                .into_iter()
                .map(|poss|
                    poss.into_iter()
                    .reduce(|prev,curr|
                        MineToClearRatio(
                            prev.0 + curr.0,
                            prev.1 + curr.1
                        )
                    ).unwrap()
                ).collect::<Vec<_>>();
            for (board_cell,refined_poss) in board.cells.iter_mut().zip(refined_possibilities) {
                match board_cell {
                    Cell::Open(_) => (),
                    Cell::Covered(ref mut prob) => *prob = refined_poss
                }
            }
        }
    }

    pub struct Border;

    impl Strategy for Border {
        fn setup(_: usize, _: usize, _: usize) -> Self {
            Border
        }
        fn refine_probabilities(board: &mut Board<Self>) {
            const THREAD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);
            let width = board.width;
            let height = board.height;
            let mines = board.mines;
            let exposed_mines = board.cells.iter()
                .fold(0,|acc,elm|
                    if matches!(elm,Cell::Open(crate::game::MCellContents::Mine)) {
                        acc + 1
                    } else { acc }
                );
            let mines_unexposed = mines - exposed_mines;
            let exposed_border = board.cells.iter().enumerate()
                .map(|(i,c)| (crate::game::Game::cell_coord(board.width, i),c))
                .filter(|(_,c)| matches!(c,Cell::Open(crate::game::MCellContents::Number(n)) if n != &0))
                .map(|(coord,_)| crate::game::Game::get_neighbors([width,height], coord))
                .flatten()
                .fold(vec![],|mut acc,coord| if acc.contains(&coord) { acc } else { acc.push(coord); acc })
                .into_iter()
                .map(|coord| (coord,&board.cells[crate::game::Game::cell_offset(width, coord)]))
                .filter(|(_,cell)| matches!(cell,Cell::Covered(_)))
                .map(|(coord,_)| coord)
                .collect::<Vec<_>>();
            let exposed_border_len = exposed_border.len();
            let (sender_perms,receiver_perms) = std::sync::mpsc::sync_channel(1);
            let exposed_border_clone = exposed_border.clone();
            std::thread::spawn(move || {
                let mut stack: Vec<Vec<([usize;2],Option<bool>)>> = vec![exposed_border_clone.into_iter().map(|coord| (coord,None)).collect::<Vec<_>>()];
                let mut out = vec![];
                while let Some(partial_perm) = stack.pop() {
                    let completed_perm_opt = partial_perm.iter()
                        .try_fold(vec![], |mut v,(coord,val)|
                            Some({
                                v.push((*coord,(*val)?));
                                v
                            })
                        );
                    if let Some(completed_perm) = completed_perm_opt {
                        out.push(completed_perm)
                    } else {
                        let existing_mines = partial_perm.iter()
                            .fold(0,|acc,elm|
                                if let (_,Some(true)) = elm { acc + 1 } else { acc }
                            );
                        if existing_mines == mines_unexposed {
                            let completed_perm = partial_perm.iter()
                                .fold(vec![],|mut acc,(coord,bool_opt)| {
                                    acc.push((*coord,(*bool_opt).unwrap_or(false)));
                                    acc
                                });
                            out.push(completed_perm);
                        } else {
                            let new_perms = vec![Some(true),Some(false)].into_iter()
                                .map(|filler_opt|
                                    partial_perm.iter()
                                        .fold((vec![],filler_opt),|(mut acc_vec,mut acc_opt),(coord,bool_opt)| {
                                            match acc_opt {
                                                Some(ref new_bool) => match bool_opt {
                                                    Some(old_bool) => acc_vec.push((*coord,Some(*old_bool))),
                                                    None => {
                                                        acc_vec.push((*coord,Some(*new_bool)));
                                                        acc_opt = None;
                                                    }
                                                },
                                                None => acc_vec.push((*coord,*bool_opt))
                                            }
                                            (acc_vec,acc_opt)
                                        }).0
                                );
                            stack.extend(new_perms);
                        }
                    }
                }
                sender_perms.send(out).unwrap();
            });
            let permutations = receiver_perms.recv_timeout(THREAD_TIMEOUT).unwrap();
            let pool = threadpool::ThreadPool::default();
            let filtered_permutations = std::sync::Arc::new(std::sync::Mutex::new(vec![]));
            for p_ext in permutations {
                let width_ext = width;
                let handle_ext = std::sync::Arc::clone(&filtered_permutations);
                let cells_ext = board.cells.clone();
                pool.execute(move || {
                    let p = p_ext;
                    let width = width_ext;
                    let handle = handle_ext;
                    let mut cells = cells_ext;
                    let cell_iter = cells.iter_mut()
                        .enumerate()
                        .map(|(i,cell)|
                            (crate::game::Game::cell_coord(width, i),cell)
                        );
                    // Define squares in permutation
                    for (coord,cell) in cell_iter {
                        if let Some((_,new_val)) = p.iter().find(|(inner_coord,_)| inner_coord == &coord) {
                            *cell = if *new_val {
                                Cell::Covered(MineToClearRatio(1,0))
                            } else {
                                Cell::Covered(MineToClearRatio(0,1))
                            };
                        }
                    }
                    // Verify game rules are not broken
                    for (coord,cell) in cells.iter().enumerate()
                        .map(|(i,cell)| (crate::game::Game::cell_coord(width, i),cell))
                    {
                        match cell {
                            Cell::Open(crate::game::MCellContents::Number(n)) => {
                                let found_count = crate::game::Game::get_neighbors([width,height], coord).into_iter()
                                    .fold(0usize,|acc,neighbor| match cells[crate::game::Game::cell_offset(width, neighbor)].classify() {
                                            CellClassification::DefinedLeft => acc + 1,
                                            _ => acc
                                        }
                                    );
                                if found_count != *n as usize {
                                    return;
                                }
                            },
                            _ => ()
                        }
                    }
                    // If all's good, add to filtered permutations
                    handle.lock().unwrap().push(p);
                })
            }
            let (sender,receiver) = std::sync::mpsc::sync_channel::<()>(2);
            let pool_handle = std::thread::spawn(move || { pool.join(); sender.send(()).unwrap(); });
            receiver.recv_timeout(THREAD_TIMEOUT).unwrap();
            let filtered_permutations_lock = filtered_permutations.lock().unwrap();
            let total_perm_count = filtered_permutations_lock.len();
            let t_permutations = filtered_permutations_lock.clone().transpose();
            let ratios_of_exposed_border = t_permutations.into_iter()
                .map(|poss| {
                    poss.into_iter().fold(0usize, |acc,(_,has_mine)| if has_mine { acc + 1 } else { acc })
                }).map(|count| MineToClearRatio(count,total_perm_count))
                .zip(exposed_border)
                .collect::<Vec<_>>();
            let overall_border_ratio = MineToClearRatio(ratios_of_exposed_border.iter().fold(0usize, |acc,elm| acc + elm.0.0),total_perm_count);
            let rest_ratio = MineToClearRatio(mines_unexposed * total_perm_count - overall_border_ratio.0,total_perm_count);
            let rest_count = board.cells.iter().filter(|c| matches!(c,Cell::Covered(_))).count() - exposed_border_len;
            let rest_cell_ratio = MineToClearRatio(rest_ratio.0,rest_ratio.1 * rest_count);

            // Apply appropriate ratios
            for (coord,cell) in board.cells.iter_mut().enumerate().map(|(i,c)| (crate::game::Game::cell_coord(width, i),c)) {
                match cell {
                    Cell::Covered(ref mut ratio) => if let Some((new_ratio,_)) = ratios_of_exposed_border.iter()
                        .find(|(_,inner_coord)| &coord == inner_coord)
                    {
                        *ratio = *new_ratio;
                    } else {
                        *ratio = rest_cell_ratio.clone();
                    },
                    _ => ()
                }
            }
            for row in board.cells.chunks(width) {
                println!("{:?}",row)
            }
        }
    }

    pub struct Random;

    impl Strategy for Random {
        fn setup(_: usize, _: usize, _: usize) -> Self {
            Random
        }
        fn refine_probabilities(board: &mut Board<Self>) {
            use std::sync::{Arc, Mutex};
            const SAMPLE_FACTOR: usize = 10;
            const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
            let width = board.width;
            let height = board.height;
            let mines = board.mines;
            let exposed_mines = board.cells.iter()
                .filter(|elm| matches!(elm,Cell::Open(crate::game::MCellContents::Mine)))
                .count();
            let unexposed_mines = mines - exposed_mines;
            let samples = board.width * board.height * SAMPLE_FACTOR;
            let permutations = Arc::new(Mutex::new(Vec::<Vec<Cell>>::with_capacity(samples)));
            let pool = threadpool::ThreadPool::default();
            let (sender,receiver) = std::sync::mpsc::sync_channel(2);
            for _ in 0..samples {
                let perm_handle = Arc::clone(&permutations);
                let cells_clone = board.cells.clone();
                pool.execute(move || {
                    use super::CellInfo;
                    use rand::Rng;
                    let mut rng = rand::thread_rng();
                    let perm_handle = perm_handle;
                    let cells_clone = cells_clone;
                    let unexposed_mines = unexposed_mines;
                    let mut out_cells = cells_clone.into_iter().map(|c| match c {
                        Cell::Open(v) => Cell::Open(v),
                        Cell::Covered(_) => Cell::Covered(MineToClearRatio(0,0))
                    }).collect::<Vec<_>>();
                    for _ in 0..unexposed_mines {
                        loop {
                            let (new_x,new_y) = (
                                rng.gen_range(0..width ),
                                rng.gen_range(0..height)
                            );
                            let new_offset = crate::game::Game::cell_offset(width, [new_x,new_y]);
                            let is_valid_place = match out_cells[new_offset] {
                                Cell::Open(_) => false,
                                Cell::Covered(_) => crate::game::Game::get_neighbors([width,height], [new_x,new_y])
                                    .into_iter()
                                    .filter_map(|coord| match out_cells[crate::game::Game::cell_offset(width, coord)] {
                                        Cell::Open(crate::game::MCellContents::Number(n)) => Some((coord,n)),
                                        _ => None
                                    }).map(|(coord,num)|
                                        crate::game::Game::get_neighbors([width,height], coord)
                                            .into_iter()
                                            .filter(|coord| matches!(
                                                out_cells[crate::game::Game::cell_offset(width,*coord)].classify(),
                                                CellClassification::DefinedLeft
                                            )).count() < num as usize
                                    ).fold(true,|acc,elm| acc && elm)
                            };
                            if is_valid_place {
                                out_cells[new_offset] = Cell::Covered(MineToClearRatio(1,0));
                                break;
                            }
                        }
                    }
                    perm_handle.lock().unwrap().push(out_cells);
                });
            }
            std::thread::spawn(move || { pool.join(); sender.send(()).unwrap() });
            receiver.recv_timeout(TIMEOUT).unwrap();
            let cloned_permutations = permutations.lock().unwrap().clone();
            let t_permutations = cloned_permutations.transpose();
            for (cell_i,cell_poss_vec) in t_permutations.into_iter().enumerate() {
                match board.cells[cell_i] {
                    Cell::Covered(ref mut ratio) => *ratio = cell_poss_vec.into_iter()
                        .map(|poss_cell| if let Cell::Covered(new_ratio) = poss_cell { new_ratio } else { panic!("Cell unexpectedly was not covered") })
                        .reduce(|prev,curr| MineToClearRatio(prev.0 + curr.0,prev.1 + curr.1))
                        .unwrap(),
                        _ => ()
                }
            }
        }
        fn best_choices(board: &Board<Self>) -> Option<(MineToClearRatio,Vec<[usize;2]>)> {
            const ERROR_RATIO_TOLERANCE: f64 = 0.01;
            let mut board_refs = board.cells.iter()
                .enumerate()
                .map(|(i,c)|
                    (crate::game::Game::cell_coord(board.width, i),c)
                ).filter_map(|(coord,c)|
                    match c {
                        Cell::Open(_) => None,
                        Cell::Covered(r) => Some((coord,r))
                    }
                ).collect::<Vec<_>>();
            board_refs.sort_unstable_by(|lhs,rhs| lhs.partial_cmp(rhs).unwrap());
            if let Some((_,&first_ratio)) = board_refs.first() {
                let best_choices = board_refs.into_iter()
                    .take_while(|(_,r)| first_ratio.clone().ratio_error(**r) < ERROR_RATIO_TOLERANCE)
                    .map(|(coord,_)| coord)
                    .collect::<Vec<_>>();
                Some((first_ratio,best_choices))
            } else { None }
        }
    }

    pub struct Localized;

    impl Strategy for Localized {
        fn setup(_: usize, _: usize, _: usize) -> Self { Localized }
        fn refine_probabilities(board: &mut Board<Self>) {
            use std::collections::HashSet;
            struct Relationship {
                max_mines: usize,
                members: Vec<[usize;2]>
            }
            #[derive(Clone)]
            struct Possibility {
                max_mines: usize,
                members: Vec<([usize;2],MineToClearRatio)>
            }
            impl From<Relationship> for Possibility {
                fn from(input: Relationship) -> Self {
                    Possibility {
                        max_mines: input.max_mines,
                        members: input.members.into_iter().map(|id| (id,MineToClearRatio(0,0))).collect::<Vec<_>>()
                    }
                }
            }
            impl std::ops::AddAssign for Relationship {
                fn add_assign(&mut self, rhs: Self) {
                    self.max_mines += rhs.max_mines;
                    self.members.extend(rhs.members);
                    self.members.sort_by(|[lhs_x,lhs_y],[rhs_x,rhs_y]|
                        match lhs_x.cmp(rhs_x) {
                            c @ (std::cmp::Ordering::Less | std::cmp::Ordering::Greater) => c,
                            std::cmp::Ordering::Equal => lhs_y.cmp(rhs_y)
                        }
                    );
                    self.members.dedup();
                }
            }
            impl std::ops::Add for Relationship {
                type Output = Self;
                fn add(mut self, rhs: Self) -> Self::Output {
                    self += rhs;
                    self
                }
            }
            let mut relationships = vec![];
            let mut covered_squares = 0usize;
            let mut uncovered_mines = 0;
            for (id,cell) in board.iter_slice(.., ..) {
                match cell {
                    Cell::Open(crate::game::MCellContents::Number(n)) => relationships.push(Relationship {
                        max_mines: *n as usize,
                        members: crate::game::Game::get_neighbors([board.width,board.height], id).into_iter()
                            .filter(|neighbor|
                                matches!(
                                    board.cells[crate::game::Game::cell_offset(board.width, *neighbor)],
                                    Cell::Covered(_)
                                )
                            ).collect::<Vec<_>>()
                    }),
                    Cell::Open(crate::game::MCellContents::Mine) => uncovered_mines += 1,
                    _ => covered_squares += 1
                }
            }
            let covered_mines = board.mines - uncovered_mines;
            let mut relationships_have_coalesced = true;
            while relationships_have_coalesced {
                relationships_have_coalesced = false;
                let relationships_iter = (0..relationships.len())
                    .flat_map(|lhs_i| (0..relationships.len())
                        .map(move |rhs_i| (lhs_i,rhs_i))
                    ).filter(|(a,b)| a < b);
                'cross_check : for (lhs_i,rhs_i) in relationships_iter {
                    let members_iter = (0..(relationships[lhs_i].members.len()))
                        .flat_map(|lhs_member_i| (0..(relationships[rhs_i].members.len()))
                            .map(move |rhs_member_i| (lhs_member_i,rhs_member_i))
                        ).filter(|(a,b)| a <= b);
                    for (lhs_member_i,rhs_member_i) in members_iter {
                        if relationships[lhs_i].members[lhs_member_i] == relationships[rhs_i].members[rhs_member_i] {
                            let popped_rhs = relationships.remove(rhs_i);
                            relationships[lhs_i] += popped_rhs;
                            relationships_have_coalesced = true;
                            break 'cross_check;
                        }
                    }
                }
            }
            let mut stack: Vec<Vec<Possibility>> = vec![relationships.into_iter().map(Possibility::from).collect::<Vec<_>>()];
            let mut finished_p_groups = vec![];
            while let Some(p_group) = stack.pop() {
                let mut p_group_is_defined = true;
                'p_groups : for (p_id,p) in p_group.iter().enumerate() {
                    if let Some((i,(_coord,_member))) = p.members.iter().enumerate()
                        .find(|(_i,(_coord,member))| matches!(member.classify(),CellClassification::FullyUndefined))
                    {
                        p_group_is_defined = false;
                        let mut mine_p_group_clone = p_group.clone();
                        mine_p_group_clone [p_id].members[i].1 = MineToClearRatio(1,0);
                        let mut clear_p_group_clone = p_group.clone();
                        clear_p_group_clone[p_id].members[i].1 = MineToClearRatio(0,1);
                        let new_p_groups = vec![mine_p_group_clone,clear_p_group_clone];

                        let p_groups_under_max = new_p_groups.into_iter()
                            .filter(|p_group| p_group.iter()
                                .all(|p| p.members.iter()
                                    .fold(0,|acc,(_cell_id,cell)|
                                        if matches!(cell.classify(),CellClassification::DefinedLeft) {
                                            acc + 1
                                        } else {
                                            acc
                                        }
                                    ) <= p.max_mines
                                )
                            ).collect::<Vec<_>>();
                        let p_groups_checked_out = p_groups_under_max.into_iter().filter(|p_group| p_group.iter()
                                .all(|p| p.members.iter()
                                    .flat_map(|(cell_id,_ratio)|
                                        crate::game::Game::get_neighbors([board.width,board.height], *cell_id)
                                    ).filter_map(|neighbor_id|
                                        match board.cells[crate::game::Game::cell_offset(board.width, neighbor_id)] {
                                            Cell::Open(crate::game::MCellContents::Number(n)) => Some((n,neighbor_id)),
                                            _ => None
                                        }
                                    ).all(|(neighbor_count,neighbor_id)| {
                                        let relevant_cell_ids = crate::game::Game::get_neighbors([board.width, board.height], neighbor_id);
                                        let mines_within_neighbor_opt = relevant_cell_ids.into_iter()
                                            .map(|relevant_cell_id| p.members.iter().find(|(p_cell_id,_)| relevant_cell_id == *p_cell_id))
                                            .map(|relevant_cell_opt| Some(matches!(
                                                relevant_cell_opt?.1.classify(),
                                                CellClassification::DefinedLeft
                                            )))
                                            .collect::<Option<Vec<_>>>().map(|bombs|
                                                bombs.into_iter()
                                                    .filter(|b| *b)
                                                    .count()
                                            );
                                        match mines_within_neighbor_opt {
                                            Some(mines_within_neighbor) => if (neighbor_count as usize) < mines_within_neighbor {
                                                false
                                            } else {
                                                true
                                            },
                                            None => true
                                        }
                                    })
                                )
                            ).collect::<Vec<_>>();
                        stack.extend(p_groups_checked_out);
                        break 'p_groups;
                    }
                }
                if p_group_is_defined {
                    let total_mines_in_p_group = p_group.iter().fold(0, |acc,p|
                            acc + p.members.iter().fold(0, |acc,(_,cell)|
                                if matches!(cell.classify(), CellClassification::DefinedLeft) {
                                    acc + 1
                                } else {
                                    acc
                                }
                            )
                        );
                    if board.mines >= total_mines_in_p_group {
                        finished_p_groups.push(p_group);
                    }
                }
            }
            let finished_board_states = finished_p_groups.into_iter().map(|p_group| {
                let merged_p_group = p_group.into_iter().flat_map(|p| p.members).collect::<Vec<_>>();
                let cells_in_p_group = merged_p_group.len();
                let covered_cells_outside_p_group = covered_squares.saturating_sub(cells_in_p_group);
                let mines_in_p_group = merged_p_group.iter().filter(|(_,r)| matches!(r.classify(),CellClassification::DefinedLeft)).count();
                let covered_mines_outside_p_group = covered_mines - mines_in_p_group;
                let covered_clears_outside_p_group = covered_cells_outside_p_group.saturating_sub(covered_mines_outside_p_group);
                let ratio_outside_clear_group = MineToClearRatio(covered_mines_outside_p_group,covered_clears_outside_p_group);
                board.iter_slice(.., ..).map(|(cell_id,cell)|
                    match cell {
                        Cell::Open(v) => Cell::Open(v.clone()),
                        Cell::Covered(_) => if let Some((_,new_ratio)) =  merged_p_group.iter().find(|(cell_id_inner,_)| &cell_id == cell_id_inner) {
                            Cell::Covered(new_ratio.clone())
                        } else {
                            Cell::Covered(ratio_outside_clear_group)
                        }
                    }
                ).collect::<Vec<_>>()
            }).collect::<Vec<_>>();
            let t_board_states = finished_board_states.transpose();
            if t_board_states.len() == 0 {
                panic!("No possibilities left")
            }
            let new_board_cells = t_board_states.into_iter().map(|cell_ratios| {
                match cell_ratios.first().unwrap() {
                    Cell::Open(v) => Cell::Open(v.clone()),
                    Cell::Covered(_) => Cell::Covered(cell_ratios.into_iter().map(|cell| match cell {
                        Cell::Covered(r) => r,
                        _ => unreachable!()
                    }).sum())
                }
            }).collect::<Vec<_>>();
            board.cells = new_board_cells;
        }
    }

}