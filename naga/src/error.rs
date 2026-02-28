use alloc::{borrow::Cow, boxed::Box, string::String};
use core::{error::Error, fmt};

#[derive(Clone, Debug)]
pub struct ShaderError<E> {
    /// The source code of the shader.
    pub source: String,
    pub label: Option<String>,
    pub inner: Box<E>,
}

#[cfg(feature = "wgsl-in")]
impl fmt::Display for ShaderError<crate::front::wgsl::ParseError> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self.label.as_deref().unwrap_or_default();
        let string = self.inner.emit_to_string(&self.source);
        write!(f, "\nShader '{label}' parsing {string}")
    }
}

#[cfg(feature = "glsl-in")]
impl fmt::Display for ShaderError<crate::front::glsl::ParseErrors> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self.label.as_deref().unwrap_or_default();
        let string = self.inner.emit_to_string(&self.source);
        write!(f, "\nShader '{label}' parsing {string}")
    }
}

#[cfg(feature = "spv-in")]
impl fmt::Display for ShaderError<crate::front::spv::Error> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self.label.as_deref().unwrap_or_default();
        let string = self.inner.emit_to_string(&self.source);
        write!(f, "\nShader '{label}' parsing {string}")
    }
}

impl fmt::Display for ShaderError<crate::WithSpan<crate::valid::ValidationError>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use codespan_reporting::{files::SimpleFile, term};

        let label = self.label.as_deref().unwrap_or_default();
        let files = SimpleFile::new(label, replace_control_chars(&self.source));
        let config = term::Config::default();

        let writer = {
            let mut writer = DiagnosticBuffer::new();
            term::emit(
                writer.inner_mut(),
                &config,
                &files,
                &self.inner.diagnostic(),
            )
            .expect("cannot write error");
            writer.into_string()
        };

        write!(f, "\nShader validation {writer}")
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "termcolor")] {
        type DiagnosticBufferInner = codespan_reporting::term::termcolor::NoColor<alloc::vec::Vec<u8>>;
        pub(crate) use codespan_reporting::term::termcolor::WriteColor as _ErrorWrite;
    } else if #[cfg(feature = "stderr")] {
        type DiagnosticBufferInner = alloc::vec::Vec<u8>;
        pub(crate) use std::io::Write as _ErrorWrite;
    } else {
        type DiagnosticBufferInner = String;
        pub(crate) use core::fmt::Write as _ErrorWrite;
    }
}

// Using this indirect export to avoid duplicating the expect(...) for all three cases above.
#[cfg_attr(
    not(any(feature = "spv-in", feature = "glsl-in")),
    expect(
        unused_imports,
        reason = "only need `ErrorWrite` with an appropriate front-end."
    )
)]
pub(crate) use _ErrorWrite as ErrorWrite;

pub(crate) struct DiagnosticBuffer {
    inner: DiagnosticBufferInner,
}

impl DiagnosticBuffer {
    #[cfg_attr(
        not(feature = "termcolor"),
        expect(
            clippy::missing_const_for_fn,
            reason = "`NoColor::new` isn't `const`, but other `inner`s are."
        )
    )]
    pub fn new() -> Self {
        cfg_if::cfg_if! {
            if #[cfg(feature = "termcolor")] {
                let inner = codespan_reporting::term::termcolor::NoColor::new(alloc::vec::Vec::new());
            } else if #[cfg(feature = "stderr")] {
                let inner = alloc::vec::Vec::new();
            } else {
                let inner = String::new();
            }
        };

        Self { inner }
    }

    pub const fn inner_mut(&mut self) -> &mut DiagnosticBufferInner {
        &mut self.inner
    }

    pub fn into_string(self) -> String {
        let Self { inner } = self;

        cfg_if::cfg_if! {
            if #[cfg(feature = "termcolor")] {
                String::from_utf8(inner.into_inner()).unwrap()
            } else if #[cfg(feature = "stderr")] {
                String::from_utf8(inner).unwrap()
            } else {
                inner
            }
        }
    }
}

// When neither `termcolor` nor `stderr` features are enabled, the
// `DiagnosticBufferInner` aliases to `String` or `Vec<u8>` earlier in this
// file.  Unfortunately `codespan_reporting::term::emit` always expects a
// writer implementing `termcolor::WriteColor`, so the default choices were
// causing the downstream build to fail with errors like:
//
// ```text
// error[E0277]: the trait bound `std::string::String: WriteColor` is not satisfied
// ```
//
// The original intent of the feature-gated type alias was to avoid pulling in
// the `termcolor` dependency when coloured output was unnecessary.  However
// `codespan-reporting` re-exports the `WriteColor` trait even without the
// `termcolor` feature, so we can provide a trivial no‑op implementation that
// simply forwards to the underlying `fmt::Write` or `io::Write` behaviour.  In
// practice the writer never needs to change colours when emitting to an
// in‑memory buffer, so the methods can all return `Ok(())` and `supports_color`
// returns `false`.
//
// We implement both `String` and `Vec<u8>` since the latter is used when the
// `stderr` feature is enabled (which also avoids a `termcolor` dependency).  By
// placing the impls in a `cfg` block we ensure they are only compiled in the
// configurations where the types actually exist.

// The implementations mirror the ones already provided for other writer types in
// the `termcolor` crate; they are intentionally small and perform no actual
// colouring.

// Note that we don't gate the module on `feature = "termcolor"` because we
// still need these impls when the feature is *disabled* (which is the only
// situation where the default aliases point at `String`/`Vec<u8>`).

#[cfg(not(feature = "termcolor"))]
mod write_color_impls {
    use super::*;
    use codespan_reporting::term::termcolor::{ColorSpec, WriteColor};
    use std::io;

    impl WriteColor for String {
        fn supports_color(&self) -> bool {
            false
        }

        fn set_color(&mut self, _spec: &ColorSpec) -> io::Result<()> {
            Ok(())
        }

        fn reset(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl WriteColor for Vec<u8> {
        fn supports_color(&self) -> bool {
            false
        }

        fn set_color(&mut self, _spec: &ColorSpec) -> io::Result<()> {
            Ok(())
        }

        fn reset(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}

impl<E> Error for ShaderError<E>
where
    ShaderError<E>: fmt::Display,
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.inner.source()
    }
}

pub(crate) fn replace_control_chars(s: &str) -> Cow<'_, str> {
    const REPLACEMENT_CHAR: &str = "\u{FFFD}";
    debug_assert_eq!(
        REPLACEMENT_CHAR.chars().next().unwrap(),
        char::REPLACEMENT_CHARACTER
    );

    let mut res = Cow::Borrowed(s);
    let mut offset = 0;

    while let Some(found_pos) = res[offset..].find(|c: char| c.is_control() && !c.is_whitespace()) {
        offset += found_pos;
        let found_len = res[offset..].chars().next().unwrap().len_utf8();
        res.to_mut()
            .replace_range(offset..offset + found_len, REPLACEMENT_CHAR);
        offset += REPLACEMENT_CHAR.len();
    }

    res
}

#[test]
fn test_replace_control_chars() {
    // The UTF-8 encoding of \u{0080} is multiple bytes.
    let input = "Foo\u{0080}Bar\u{0001}Baz\n";
    let expected = "Foo\u{FFFD}Bar\u{FFFD}Baz\n";
    assert_eq!(replace_control_chars(input), expected);
}
