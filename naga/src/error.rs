// We deliberately continue to use `#![no_std]` for most of the crate,
// but our diagnostics code needs access to the standard library in order to
// satisfy the various writer traits that `codespan_reporting::term::emit`
// may demand.  `extern crate std` brings the `std` crate into scope even when
// `#![no_std]` is present; the path `std::io` can then be referenced without
// `#[cfg]` guards.
extern crate std;

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
            let mut w = writer.writer();
            term::emit(
                &mut w,
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
        type DiagnosticBufferInner = termcolor::NoColor<alloc::vec::Vec<u8>>;
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
                let inner = termcolor::NoColor::new(alloc::vec::Vec::new());
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


impl<E> Error for ShaderError<E>
where
    ShaderError<E>: fmt::Display,
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.inner.source()
    }
}

/// Adapt the internal buffer to whatever writer interface `codespan_reporting`
/// requires.  The diagnostics crate's `term::emit` function is compiled with
/// different signatures depending on its features; we make our wrapper
/// implement *all* of the possibilities so that callers can always pass it
/// directly without worrying about which one will be chosen.
#[derive(Debug)]
pub(crate) struct DiagnosticBufferWriter<'a> {
    inner: &'a mut DiagnosticBufferInner,
}

impl<'a> DiagnosticBufferWriter<'a> {
    pub(crate) fn new(inner: &'a mut DiagnosticBufferInner) -> Self {
        Self { inner }
    }
}

// `termcolor` is pulled in either via naga's own feature or indirectly when
// *codespan-reporting* enables its `termcolor` feature.  Having it as a normal
// dependency means the trait is always in scope and we can implement it
// unconditionally.
use termcolor::{ColorSpec, WriteColor};

impl<'a> WriteColor for DiagnosticBufferWriter<'a> {
    fn supports_color(&self) -> bool {
        false
    }

    fn set_color(&mut self, _spec: &ColorSpec) -> std::io::Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// `std::io::Write` may be required even in a `#![no_std]` build because
// `codespan-reporting` could be compiled with `std`.  We unconditionally
// implement the trait and perform conversions depending on what the inner
// buffer actually is.
impl<'a> std::io::Write for DiagnosticBufferWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // The concrete type of `DiagnosticBufferInner` depends on which features are
        // active.  When `termcolor` is present (regardless of `stderr`) it is
        // `NoColor<Vec<u8>>`.  Otherwise `stderr` yields `Vec<u8>` and the
        // no‑feature case uses `String`.  We use `cfg_if` to handle each case
        // correctly without duplicating the outer logic.
        cfg_if::cfg_if! {
            if #[cfg(feature = "termcolor")] {
                // inner is NoColor<Vec<u8>>
                self.inner.get_mut().extend_from_slice(buf);
            } else if #[cfg(feature = "stderr")] {
                // inner is Vec<u8>
                self.inner.extend_from_slice(buf);
            } else {
                // inner is String
                let s = std::str::from_utf8(buf)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                self.inner.push_str(s);
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// Finally, support the fmt::Write case for true no-std builds with neither
// `std` nor `termcolor` available.
impl<'a> fmt::Write for DiagnosticBufferWriter<'a> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        cfg_if::cfg_if! {
            if #[cfg(feature = "termcolor")] {
                self.inner.get_mut().extend_from_slice(s.as_bytes());
            } else if #[cfg(feature = "stderr")] {
                self.inner.extend_from_slice(s.as_bytes());
            } else {
                self.inner.push_str(s);
            }
        }
        Ok(())
    }
}

impl DiagnosticBuffer {
    /// Create a writer adaptor suitable for passing to
    /// `codespan_reporting::term::emit`.
    pub fn writer(&mut self) -> DiagnosticBufferWriter<'_> {
        DiagnosticBufferWriter::new(self.inner_mut())
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
