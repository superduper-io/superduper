#!/usr/bin/env bash

set -x

./k6 run --out dashboard=open qtd_stress.js
