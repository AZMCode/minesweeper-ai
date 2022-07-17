use crate::utils::prelude::*;
use eframe::egui;

const RED: egui::Color32 = egui::Color32::DARK_RED;
const DARK: egui::Color32 = egui::Color32::DARK_GRAY;
const BLACK: egui::Color32 = egui::Color32::BLACK;
const WHITE: egui::Color32 = egui::Color32::WHITE;
const BLANK_TEXT: &'static str = "  ";


pub struct App {
    ai: crate::ai::AnyBoard,
    show_ai_moves: bool,
    field_size: [usize;2],
    field_size_input: [String;2],
    num_mines: usize,
    num_mines_input: String,
    board_kind: String,
    field_ui: FieldUI
}

impl App {
    pub fn new() -> Result<Self> {
        const GAME_PARAMS: [usize;3] = [5,5,5];
        let field_ui = FieldUI::new(GAME_PARAMS[0], GAME_PARAMS[1], GAME_PARAMS[2])?;
        println!("Calculating initial Board");
        let ai = crate::ai::AnyBoard::new("Localized",GAME_PARAMS[0], GAME_PARAMS[1], GAME_PARAMS[2], field_ui.field.iter().map(|(_,c)| c))?;
        println!("Finished calculating initial board");
        Ok(App {
            field_ui, ai,
            field_size: [GAME_PARAMS[0],GAME_PARAMS[1]],
            field_size_input: [GAME_PARAMS[0].to_string(),GAME_PARAMS[1].to_string()],
            num_mines: GAME_PARAMS[2],
            num_mines_input: GAME_PARAMS[2].to_string(),
            board_kind: "Localized".to_string(),
            show_ai_moves: false
        })
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            use egui::Widget;
            let mut field_params_have_changed = false;
            ui.horizontal(|inner_ui| {
                inner_ui.label("Dimensions (Width x Height):");
                self.field_size_input = self.field_size_input.clone().map(|mut s| {
                    let mut edit = egui::TextEdit::singleline(&mut s);
                    edit = edit.desired_width(30.0);
                    inner_ui.add(edit);
                    s
                });
                match self.field_size_input.clone().try_map(|s| s.trim().parse::<usize>()) {
                    Ok(v) => if self.field_size != v {
                        self.field_size = v;
                        field_params_have_changed = true;
                    },
                    _ => ()
                }
            });
            ui.horizontal(|inner_ui|{
                inner_ui.label("Number of mines:");
                egui::TextEdit::singleline(&mut self.num_mines_input).desired_width(30.0).ui(inner_ui);
                match self.num_mines_input.trim().parse::<usize>() {
                    Ok(v) => if self.num_mines != v {
                        self.num_mines = v;
                        field_params_have_changed = true;
                    },
                    _ => ()
                };
            });

            if field_params_have_changed {
                self.field_ui = FieldUI::new(self.field_size[0], self.field_size[1], self.num_mines).unwrap();
                self.ai = crate::ai::AnyBoard::new(
                    &self.board_kind,
                    self.field_size[0],
                    self.field_size[1],
                    self.num_mines,
                    self.field_ui.field.iter().map(|(_,c)| c)
                ).unwrap();
            }

            let mut new_board_kind = self.board_kind.clone();
            ui.horizontal(|inner_ui| {
                inner_ui.label("Board Kind:");
                let mut combo_box = egui::ComboBox::new("board_kind", "");
                combo_box = combo_box.selected_text(&new_board_kind);
                match combo_box.show_ui(inner_ui, |combo_ui| {
                    crate::ai::STRATEGY_NAMES.iter()
                        .map(|s| combo_ui.button(s.to_string()).clicked())
                        .enumerate()
                        .find(|(i,v)| *v)
                        .map(|(i,_)| i)
                }).inner {
                    Some(Some(i)) => new_board_kind = crate::ai::STRATEGY_NAMES[i].to_string(),
                    _ => ()
                }
            });
            
            if self.board_kind != new_board_kind {
                self.board_kind = new_board_kind;
                self.ai = crate::ai::AnyBoard::new(
                    &self.board_kind,
                    self.field_size[0],
                    self.field_size[1],
                    self.num_mines,
                    self.field_ui.field.iter().map(|(_,c)| c)
                ).unwrap();
            }
            
            if ui.button("New Game").clicked() {
                self.field_ui = FieldUI::new(self.field_size[0], self.field_size[1], self.num_mines).unwrap();
                self.ai = crate::ai::AnyBoard::new(
                    &self.board_kind,
                    self.field_size[0],
                    self.field_size[1],
                    self.num_mines,
                    self.field_ui.field.iter().map(|(_,c)| c)
                ).unwrap();
            }

            self.show_ai_moves =  if ui.button(
                if self.show_ai_moves { "Hide AI Moves" } else { "Show AI Moves" }
            ).clicked() { !self.show_ai_moves } else { self.show_ai_moves };
            self.field_ui.highlight_coords = if self.show_ai_moves {
                Some(self.ai.pick_move())
            } else {
                None
            };
            let out = (&mut self.field_ui).ui(ui);
            if self.field_ui.field_changed {
                println!("Calculating Board Update");
                self.ai.update_board(self.field_ui.field.iter().map(|(_,c)| c));
                println!("Finished Calculating Board Update");
            }
            out
        });
    }
}


pub struct FieldUI {
    field: crate::game::Game,
    pub highlight_coords: Option<crate::ai::MoveType>,
    field_changed: bool,
    flagged: Vec<bool>
}

impl FieldUI {
    pub fn new(width: usize, height: usize, mines: usize) -> Result<Self>{
        Ok(FieldUI {
            field: crate::game::Game::new(width,height,mines)?,
            highlight_coords: None,
            field_changed: true,
            flagged: vec![false;width*height]
        })
    }
}

impl<'f> egui::Widget for &'f mut FieldUI {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        self.field_changed = false;
        enum ClickType { Primary, Secondary }
        let mut coord_clicked = None;
        let [cells_x,_] = self.field.dimensions();
        let curr_cell_states = self.field.iter().enumerate();
        let highlight_coords = self.highlight_coords.clone();
        let table_inner_res = egui::Grid::new(egui::Id::new("Minesweeper Field")).show(ui, |table_ui| 
            curr_cell_states
                .map(|(offset,(coord,cell))| {
                    let out = (coord,CellUI::new(&cell,*(self.flagged.get(offset).unwrap()),
                        match &highlight_coords {
                            Some(crate::ai::MoveType::Certain(h_cells))          if h_cells.contains(&coord) => CellHighlighting::Certain,
                            Some(crate::ai::MoveType::BestChanceRandom(h_cells)) if h_cells.contains(&coord) => CellHighlighting::Random ,
                            _                                                                                => CellHighlighting::NoHighlighting
                        }
                    ).ui(table_ui));
                    if offset % cells_x == cells_x - 1 {
                        table_ui.end_row();
                    }
                    out
                }).map(|(coord,resp)| {
                    if resp.clicked_by(egui::PointerButton::Primary) {
                        coord_clicked = Some((ClickType::Primary,coord));
                    } else if resp.clicked_by(egui::PointerButton::Secondary) {
                        coord_clicked = Some((ClickType::Secondary,coord))
                    }
                    resp
                }).reduce(|prev,curr| prev.union(curr))
                .unwrap()
        );
        if let Some((kind,coord)) = coord_clicked {
            if let ClickType::Primary = kind {
                match self.field.iter().enumerate().find(|(_,(i_coord,_))| i_coord == &coord).unwrap() {
                    (offset,(_,crate::game::MCellHandle::Covered)) if !self.flagged.get(offset).unwrap() => {
                        self.field_changed = true;
                        self.field.uncover(coord);
                    },
                    _ => ()
                }
            } else {
                match self.field.iter().enumerate().find(|(_,(i_coord,_))| i_coord == &coord).unwrap() {
                    (offset,(_,crate::game::MCellHandle::Covered)) => {
                        let curr_flag_ref = self.flagged.get_mut(offset).unwrap();
                        *curr_flag_ref = !*curr_flag_ref;
                    },
                    _ => ()
                }
            }
        }
        table_inner_res.inner
    }
}

enum CellHighlighting {
    Certain,
    Random,
    NoHighlighting
}

struct CellUI<'r,'c> {
    cell_handle: &'r crate::game::MCellHandle<'c>,
    is_flagged: bool,
    highlighting: CellHighlighting
}

impl<'r,'c> CellUI<'r,'c> {
    pub fn new(cell_handle: &'r crate::game::MCellHandle<'c>, is_flagged: bool, highlighting: CellHighlighting) -> Self {
        CellUI { cell_handle, is_flagged, highlighting }
    }
}

impl<'r,'c> egui::Widget for CellUI<'r,'c> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let bg_color = match self.highlighting {
            CellHighlighting::Certain => Some(egui::Color32::GREEN ),
            CellHighlighting::Random  => Some(egui::Color32::YELLOW),
            CellHighlighting::NoHighlighting => None
        };
        if self.is_flagged {
            ui.add(
                egui::Button::new(
                    egui::WidgetText::RichText(
                        egui::RichText::from("F")
                            .color(RED)
                    )
                ).fill(bg_color.unwrap_or(DARK))
                .sense(egui::Sense::click())
            )
        } else {
            match self.cell_handle {
                crate::game::MCellHandle::Covered => {
                    ui.add(
                        egui::Button::new(BLANK_TEXT)
                            .fill(bg_color.unwrap_or(DARK))
                            .sense(egui::Sense::click())
                            .wrap(true)
                    )
                },
                crate::game::MCellHandle::Open(info) => match info {
                    crate::game::MCellContents::Mine => ui.add(
                        egui::Button::new(
                            egui::WidgetText::RichText(
                                egui::RichText::new("M")
                                    .color(WHITE)
                            )
                        )
                            .fill(RED)
                            .sense(egui::Sense::focusable_noninteractive())
                            .wrap(true)
                    ),
                    crate::game::MCellContents::Number(n) => ui.add(
                        egui::Button::new(egui::WidgetText::RichText(
                            egui::RichText::from(if n == &0 { BLANK_TEXT.to_string() } else { n.to_string() })
                                .color(BLACK)
                        ))
                            .fill(WHITE)
                            .sense(egui::Sense::focusable_noninteractive())
                            .wrap(true)
                    )
                }
            }
        }
    }
}
