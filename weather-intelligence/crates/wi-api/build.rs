fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir(&out_dir)
        .compile_protos(&["proto/weather.proto"], &["proto"])?;

    println!("cargo:rerun-if-changed=proto/weather.proto");

    Ok(())
}
