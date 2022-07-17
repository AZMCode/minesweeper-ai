#[derive(Debug,thiserror::Error)]
pub enum Error {
    #[error("Error during creation of Game: {0}")]
    GameError(#[from] crate::game::GameError),
    #[error("Error during running of AI system")]
    AIError(#[from] crate::ai::AIError)
}

pub type Result<T = (), E = Error> = std::result::Result<T,E>;