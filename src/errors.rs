//! Errors returned by the public API and internal validation.

use std::io;
use std::path::{Path, PathBuf};

use thiserror::Error;

/// Failures exposed by the library.
#[derive(Debug, Error)]
pub enum TicitError {
    /// A caller supplied an invalid size, record id, mask, or other value.
    #[error("{message}")]
    InvalidInput {
        /// Human-readable validation failure.
        message: String,
    },

    /// Circuit source is syntactically malformed.
    #[error("{message}")]
    Parse {
        /// One-based source line where parsing failed.
        line: usize,
        /// Human-readable parse failure, including the line number.
        message: String,
    },

    /// A circuit file could not be read.
    #[error("{message}")]
    Io {
        /// Path ticit attempted to read.
        path: PathBuf,
        /// Stable high-level error message.
        message: String,
        /// Original operating-system I/O error.
        #[source]
        source: io::Error,
    },

    /// The input uses a valid construct that ticit does not implement.
    #[error("{message}")]
    Unsupported {
        /// Human-readable unsupported-feature description.
        message: String,
    },

    /// An internal invariant failed without panicking.
    #[error("{message}")]
    Internal {
        /// Human-readable invariant failure.
        message: String,
    },

    /// A scoped CPU sampling worker panicked.
    #[error("batch worker panicked")]
    WorkerPanic,
}

impl PartialEq for TicitError {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
            && self.message() == other.message()
    }
}

impl Eq for TicitError {}

impl TicitError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    pub(crate) fn parse(line: usize, detail: impl std::fmt::Display) -> Self {
        Self::Parse {
            line,
            message: format!("line {line}: {detail}"),
        }
    }

    pub(crate) fn io(path: &Path, source: io::Error) -> Self {
        Self::Io {
            message: format!("failed to read circuit file: {}", path.to_string_lossy()),
            path: path.to_owned(),
            source,
        }
    }

    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Returns the stable human-readable message carried by this error.
    ///
    /// This is the same text emitted by [`std::fmt::Display`], but borrowing it
    /// avoids allocating a `String` when an embedding API maps errors.
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::InvalidInput { message }
            | Self::Parse { message, .. }
            | Self::Io { message, .. }
            | Self::Unsupported { message }
            | Self::Internal { message } => message,
            Self::WorkerPanic => "batch worker panicked",
        }
    }
}

/// Result type used by ticit's parsing, preparation, and CPU sampling APIs.
pub type Result<T> = std::result::Result<T, TicitError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_variants_preserve_messages_and_context() {
        let error = TicitError::parse(7, "boom");
        assert_eq!(error.to_string(), "line 7: boom");
        assert_eq!(error.message(), "line 7: boom");
        assert!(matches!(error, TicitError::Parse { line: 7, .. }));
    }
}
