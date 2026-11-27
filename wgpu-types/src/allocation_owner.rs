//! HELIO PATCH: who allocated GPU memory.
//!
//! An application marks the stretch of code that runs for one owner of GPU
//! memory with [`scope`]; every allocation the backends make on this thread
//! while the scope lives is named `owner<US>label` in the allocator report
//! ([`crate::AllocatorReport`]), where `<US>` is [`SEPARATOR`]. Scopes nest:
//! the owner is the scopes' names joined by `/`, outermost first. A
//! [`root_scope`] starts a new owner path whatever encloses it: a library that
//! owns what it allocates (a scene database called from a renderer) names
//! itself, not its caller. An allocation made outside any scope keeps its
//! plain label.
//!
//! This is what lets a memory tool attribute every byte to its owner without
//! every call that creates a buffer or texture naming one: the owner is set
//! once where the work for it starts. [`split`] takes a report name apart.

use alloc::{borrow::Cow, string::String, vec::Vec};
use core::cell::RefCell;
use core::marker::PhantomData;

/// Separates the owner from the label in an allocation name (ASCII unit
/// separator: never part of a label anyone types).
pub const SEPARATOR: char = '\u{1f}';

struct Entry {
    owner: Cow<'static, str>,
    root: bool,
}

std::thread_local! {
    static OWNERS: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
}

/// The owner of this thread's GPU allocations until it is dropped.
#[must_use = "the scope ends when this guard is dropped"]
#[derive(Debug)]
pub struct OwnerScope {
    // Not `Send`: the scope is the creating thread's.
    _thread: PhantomData<*const ()>,
}

fn push(owner: Cow<'static, str>, root: bool) -> OwnerScope {
    OWNERS.with(|owners| owners.borrow_mut().push(Entry { owner, root }));
    OwnerScope {
        _thread: PhantomData,
    }
}

/// Name `owner`, under the enclosing scopes, as the owner of every GPU
/// allocation this thread makes until the returned guard drops.
pub fn scope(owner: impl Into<Cow<'static, str>>) -> OwnerScope {
    push(owner.into(), false)
}

/// Like [`scope`], but `owner` heads the path: the enclosing scopes are not
/// part of it until the guard drops.
pub fn root_scope(owner: impl Into<Cow<'static, str>>) -> OwnerScope {
    push(owner.into(), true)
}

impl Drop for OwnerScope {
    fn drop(&mut self) {
        OWNERS.with(|owners| {
            owners.borrow_mut().pop();
        });
    }
}

/// This thread's current owner (`None` outside any scope).
pub fn current() -> Option<String> {
    OWNERS.with(|owners| {
        let owners = owners.borrow();
        let start = owners.iter().rposition(|entry| entry.root).unwrap_or(0);
        let path = &owners[start..];
        (!path.is_empty()).then(|| {
            let mut joined = String::new();
            for (index, entry) in path.iter().enumerate() {
                if index > 0 {
                    joined.push('/');
                }
                joined.push_str(&entry.owner);
            }
            joined
        })
    })
}

/// The name a backend gives the allocation for a resource labelled `label`.
pub fn allocation_name(label: &str) -> Cow<'_, str> {
    match current() {
        Some(owner) => {
            let mut name = owner;
            name.push(SEPARATOR);
            name.push_str(label);
            Cow::Owned(name)
        }
        None => Cow::Borrowed(label),
    }
}

/// An allocation name taken apart: its owner (when it was made in a scope) and
/// its label.
pub fn split(name: &str) -> (Option<&str>, &str) {
    match name.split_once(SEPARATOR) {
        Some((owner, label)) => (Some(owner), label),
        None => (None, name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_nest_and_end_with_their_guard() {
        assert_eq!(allocation_name("buf"), "buf");
        {
            let _outer = scope("Helio");
            {
                let _inner = scope("Shadow");
                assert_eq!(split(&allocation_name("atlas")), (Some("Helio/Shadow"), "atlas"));
            }
            assert_eq!(current().as_deref(), Some("Helio"));
        }
        assert_eq!(current(), None);
        assert_eq!(split("plain"), (None, "plain"));
    }

    #[test]
    fn a_root_scope_heads_the_path_until_it_ends() {
        let _helio = scope("Helio");
        {
            let _scenedb = root_scope("SceneDB");
            let _pool = scope(String::from("pool"));
            assert_eq!(current().as_deref(), Some("SceneDB/pool"));
        }
        assert_eq!(current().as_deref(), Some("Helio"));
    }
}
