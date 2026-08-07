{ pkgs ? import <nixpkgs> { } }:

pkgs.rustPlatform.buildRustPackage {
  pname = "fvnn-mini-nn-verifier";
  version = "0.0.1";

  src = ./.;

  cargoLock.lockFile = ./Cargo.lock;

  # Uncomment if a build ever needs a C compiler / system libs.
  # nativeBuildInputs = [ pkgs.pkg-config ];
  # buildInputs = [ ];

  meta = with pkgs.lib; {
    description = "FVNN mini neural-network verifier";
    mainProgram = "FVNN-Mini-NN-Verifier";
    platforms = platforms.all;
  };
}
