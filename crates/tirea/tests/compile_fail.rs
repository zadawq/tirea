#[cfg(not(feature = "ag-ui"))]
#[test]
fn tirea_does_not_reexport_ag_ui_protocol_types() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/no_agui_reexports.rs");
}

#[cfg(feature = "ag-ui")]
#[test]
fn tirea_reexports_ag_ui_protocol_module_when_enabled() {
    let _ = std::mem::size_of::<tirea::ag_ui::Event>();
}
