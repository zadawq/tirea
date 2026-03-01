#[test]
fn lattice_compile_errors() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/lattice_*.rs");
}
