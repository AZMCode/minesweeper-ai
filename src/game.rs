use crate::utils::prelude::*;

#[derive(Debug,Clone)]

pub enum MCellContents {
    Number(u8),
    Mine
}

#[derive(Debug,Clone)]
enum MCellState {
    Covered,
    Open
}

#[derive(Debug)]
pub enum MCellHandle<'c> {
    Covered,
    Open(&'c MCellContents)
}

impl<'c> MCellHandle<'c> {
    pub fn clone_out(&self) -> MCell {
        match self {
            MCellHandle::Covered => MCell::Covered,
            MCellHandle::Open(contents) => MCell::Open(contents.clone().clone())
        }
    }
}

impl<'c> From<MCellHandle<'c>> for MCell {
    fn from(input: MCellHandle<'c>) -> Self {
        input.clone_out()
    }
}

#[derive(Debug)]
pub enum MCell {
    Covered,
    Open(MCellContents)
}

#[derive(Debug, Clone)]
struct InternalMCell {
    state: MCellState,
    contents: MCellContents
}

#[derive(Debug, Clone)]
pub enum GameState {
    Running,
    Lost,
    Won
}

#[derive(Debug)]
enum GameContents {
    Undefined,
    Defined {
        game_state: GameState,
        cells: Vec<InternalMCell>
    },
}

pub struct Game {
    contents: GameContents,
    width: usize,
    height: usize,
    mines: usize
}

#[derive(Debug,thiserror::Error)]
pub enum GameError {
    #[error("Tried to create a Game ( with {requested} spaces ) with too many mines ( {available} )")]
    TooManyMines {
        requested: usize,
        available: usize
    },
    #[error("Tried to create a Game with zero cells ( {width} x {height} )")]
    ZeroAreaBoard { width: usize, height: usize}
}

impl Game {
    pub fn new(width: usize, height: usize, mines: usize) -> Result<Self,GameError> {
        if height == 0 || width == 0 {
            return Err(GameError::ZeroAreaBoard { width, height })
        }
        if mines > (width * height) {
            return Err(GameError::TooManyMines { requested: mines, available: width * height })
        }
        Ok(Game {
            width, height, mines,
            contents: GameContents::Undefined
        })
    }
    pub fn iter(&self) -> GameIter<'_> {
        <&'_ Self as std::iter::IntoIterator>::into_iter(self)
    }
    pub fn dimensions(&self) -> [usize;2] {
        [self.width,self.height]
    }
    pub fn cell_offset(width: usize, coords: [usize;2]) -> usize {
        (width * coords[1]) + coords[0]
    }
    pub fn cell_coord(width: usize, offset: usize) -> [usize;2] {
        [offset % width,(offset / width)]
    }
    pub fn get_neighbors(size: [usize;2], coord: [usize;2]) -> Vec<[usize;2]> {
        let mut out = Vec::with_capacity(8);
        let at_top = coord[1] == 0;
        let at_bottom = coord[1] == (size[1] - 1);
        let at_left = coord[0] == 0;
        let at_right = coord[0] == size[0] - 1;
        if !at_top && !at_left {
            out.push([ coord[0] - 1, coord[1] - 1 ]);
        }
        if !at_top {
            out.push([ coord[0]    , coord[1] - 1 ]);
        }
        if !at_top && !at_right {
            out.push([ coord[0] + 1, coord[1] - 1 ]);
        }
        if !at_left {
            out.push([ coord[0] - 1, coord[1]     ]);
        }
        if !at_right {
            out.push([ coord[0] + 1, coord[1]     ]);
        }
        if !at_bottom && !at_left {
            out.push([ coord[0] - 1, coord[1] + 1 ]);
        }
        if !at_bottom {
            out.push([ coord[0]    , coord[1] + 1 ]);
        }
        if !at_bottom && !at_right {
            out.push([ coord[0] + 1, coord[1] + 1 ]);
        }
        out
    }
    fn define_contents(&mut self, initial_uncovering: [usize;2]) {
        self.contents = GameContents::Defined { game_state: GameState::Running,
            cells: {
                let size = self.width * self.height;
                let mut out = vec![
                    InternalMCell {
                        contents: MCellContents::Number(0),
                        state: MCellState::Covered
                    }; size
                ];
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let random_coords_generator =
                    (|| Some([
                        rng.gen_range(0..self.width),
                        rng.gen_range(0..self.height)
                    ])).generate_iter();
                let random_unique_noninitial_offsets = random_coords_generator
                    .scan(vec![initial_uncovering.clone()], |acc,curr_coord| {
                        if !acc.contains(&curr_coord) { acc.push(curr_coord) }
                        Some(acc.clone())
                    }).skip_while(|v| v.len() != (self.mines + 1))
                    .take(1)
                    .collect::<Vec<_>>()[0].clone()
                    .into_iter()
                    .filter(|elm| elm != &initial_uncovering)
                    .map(|coord| Self::cell_offset(self.width,coord))
                    .collect::<Vec<_>>();
                for offset in random_unique_noninitial_offsets {
                    out[offset].contents = MCellContents::Mine;
                }
                let coordinates = (0..self.width).map(|x| (0..self.height).map(move |y| [x,y])).flatten();
                for coord in coordinates {
                    if let InternalMCell { contents: MCellContents::Number(_), .. } = out[Self::cell_offset(self.width,coord)].clone() {
                        let mut neighbor_count = 0;
                        for neighbor in Self::get_neighbors([self.width,self.height],coord) {
                            if let InternalMCell { contents: MCellContents::Mine, .. } = out[Self::cell_offset(self.width,neighbor)] {
                                neighbor_count += 1;
                            }
                        }
                        if neighbor_count != 0 {
                            out[Self::cell_offset(self.width,coord)].contents = MCellContents::Number(neighbor_count);
                        }
                    }
                }
                out
            }
        };
        self.normalize_board();
    }
    fn normalize_board(&mut self) -> GameState {
        match self.contents {
            GameContents::Undefined => self.calculate_game_state(),
            GameContents::Defined { ref mut cells, ref mut game_state } => {
                let mut changes_may_be_needed = true;
                let size = self.width * self.height;
                while changes_may_be_needed {
                    changes_may_be_needed = false;
                    for offset in 0..size {
                        if let InternalMCell { state: MCellState::Open, contents: MCellContents::Number(0) } = cells[offset] {
                            for neighbor in Self::get_neighbors([self.width,self.height],Self::cell_coord(self.width,offset)) {
                                if let ref mut state @ MCellState::Covered = cells[Self::cell_offset(self.width,neighbor)].state {
                                    changes_may_be_needed = true;
                                    *state = MCellState::Open;
                                }
                            }
                        }
                    }
                }
                *game_state = Self::calculate_game_state_defined(cells);
                game_state.clone()
            }
        }
    }
    fn calculate_game_state(&self) -> GameState {
        match self.contents {
            GameContents::Undefined => if self.mines == (self.width * self.height) { GameState::Won } else { GameState::Running },
            GameContents::Defined { ref cells, .. } => Self::calculate_game_state_defined(cells)
        }
    }
    fn calculate_game_state_defined(cells: &[InternalMCell]) -> GameState {
        let mut is_complete = true;
        for cell in cells.iter() {
            match cell {
                InternalMCell { contents: MCellContents::Mine,      state: MCellState::Open    } => return GameState::Lost,
                InternalMCell { contents: MCellContents::Number(_), state: MCellState::Covered } => is_complete = false,
                _ => ()
            }
        }
        if is_complete {
            GameState::Won
        } else {
            GameState::Running
        }
    }
    pub fn uncover(&mut self, coord: [usize;2]) -> GameState {
        match self.contents {
            GameContents::Undefined => {
                self.define_contents(coord);
                self.uncover(coord)
            },
            GameContents::Defined { ref mut cells, .. } => {
                cells[Self::cell_offset(self.width,coord)].state = MCellState::Open;
                self.normalize_board()
            }
        }
    }
}

pub struct GameIter<'g>{
    game_ref: &'g Game,
    index: usize
}

impl<'g> Iterator for GameIter<'g> {
    type Item = ([usize;2],MCellHandle<'g>);
    fn next(&mut self) -> Option<Self::Item> {
        let size = self.game_ref.width * self.game_ref.height;
        if self.index >= size {
            None
        } else {
            let width = self.game_ref.width;
            let out = match self.game_ref.contents {
                GameContents::Undefined => Some((Game::cell_coord(width, self.index), MCellHandle::Covered)),
                GameContents::Defined { ref cells, .. } => {
                    let InternalMCell { ref state, ref contents } = cells.get(self.index).unwrap();
                    Some((
                        Game::cell_coord(width, self.index),
                        match state {
                            MCellState::Covered => MCellHandle::Covered,
                            MCellState::Open    => MCellHandle::Open(contents)
                        }
                    ))
                }
            };
            self.index += 1;
            out
        }
    }
}

impl<'g> std::iter::IntoIterator for &'g Game {
    type IntoIter = GameIter<'g>;
    type Item = ([usize;2],MCellHandle<'g>);
    fn into_iter(self) -> Self::IntoIter {
        GameIter {
            game_ref: self,
            index: 0
        }
    }
}
