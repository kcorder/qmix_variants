#!/bin/bash 

# Tar-gzip the results directory, but exclude all checkpoints
tar --exclude='*checkpoint*' -czvf results.tar.gz results/
