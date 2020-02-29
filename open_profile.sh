gprof2dot -f pstats profile.pstats | dot -Tpdf -o callgraph.pdf && chromium callgraph.pdf
