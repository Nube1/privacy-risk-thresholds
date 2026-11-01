# Cryptography policy of the disclosure-risk engine

1. The engine contains no cryptography of its own. Decision records carry SHA-256 digests,
   which detect change and say nothing about who made a record.
2. Signing records and protecting the register and its transport belong to the institution's
   security programme, using maintained implementations of FIPS 203 (ML-KEM), FIPS 204
   (ML-DSA) and FIPS 205 (SLH-DSA) through supported protocols.
3. Bespoke ciphers, including chaotic ciphers, are not to be added.
4. Keys and randomness are never derived from model trajectories (for example a Lorenz
   trajectory). Numerical unpredictability is not cryptographic entropy.
5. Quantum attack coverage on released data is recorded as unresolved until an implemented
   attack or a defensible resource analysis exists.
