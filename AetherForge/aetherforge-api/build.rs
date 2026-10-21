fn main() {
    // Use prost-build for protobuf compilation (stable API)
    if let Err(e) = prost_build::Config::new()
        .out_dir(std::env::var("OUT_DIR").unwrap())
        .compile_protos(&["proto/aetherforge.proto"], &["proto"])
    {
        eprintln!("Failed to compile prost: {}", e);
        std::process::exit(1);
    }
    
    // tonic-build 0.13 API changed - we use prost-build only
    // The generated code from prost_build will be used by tonic
    
    println!("cargo:rerun-if-changed=proto/aetherforge.proto");
}
