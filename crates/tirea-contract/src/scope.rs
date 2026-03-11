use crate::runtime::ScopePolicy;

/// Check whether an identifier is allowed by optional allow/deny lists.
#[must_use]
pub fn is_id_allowed(id: &str, allowed: Option<&[String]>, excluded: Option<&[String]>) -> bool {
    if let Some(allowed) = allowed {
        if !allowed.iter().any(|value| value == id) {
            return false;
        }
    }
    if let Some(excluded) = excluded {
        if excluded.iter().any(|value| value == id) {
            return false;
        }
    }
    true
}

/// Scope domains carried in [`ScopePolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeDomain {
    Tool,
    Skill,
    Agent,
}

/// Check whether an identifier is allowed by the typed policy for a scope domain.
#[must_use]
pub fn is_scope_allowed(policy: Option<&ScopePolicy>, id: &str, domain: ScopeDomain) -> bool {
    let (allowed, excluded) = match domain {
        ScopeDomain::Tool => (
            policy.and_then(ScopePolicy::allowed_tools),
            policy.and_then(ScopePolicy::excluded_tools),
        ),
        ScopeDomain::Skill => (
            policy.and_then(ScopePolicy::allowed_skills),
            policy.and_then(ScopePolicy::excluded_skills),
        ),
        ScopeDomain::Agent => (
            policy.and_then(ScopePolicy::allowed_agents),
            policy.and_then(ScopePolicy::excluded_agents),
        ),
    };
    is_id_allowed(id, allowed, excluded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_id_allowed_uses_allow_and_exclude_lists() {
        let allowed = vec!["a".to_string(), "b".to_string()];
        let excluded = vec!["b".to_string()];
        assert!(is_id_allowed("a", Some(&allowed), Some(&excluded)));
        assert!(!is_id_allowed("b", Some(&allowed), Some(&excluded)));
        assert!(!is_id_allowed("c", Some(&allowed), Some(&excluded)));
    }

    #[test]
    fn is_scope_allowed_reads_tool_filters_from_policy() {
        let mut policy = ScopePolicy::new();
        policy.set_allowed_tools_if_absent(Some(&["a".to_string(), "b".to_string()]));
        policy.set_excluded_tools_if_absent(Some(&["b".to_string()]));

        assert!(is_scope_allowed(Some(&policy), "a", ScopeDomain::Tool));
        assert!(!is_scope_allowed(Some(&policy), "b", ScopeDomain::Tool));
        assert!(!is_scope_allowed(Some(&policy), "c", ScopeDomain::Tool));
    }
}
