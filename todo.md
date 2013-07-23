- fix wiki pages
- move wiki to pages (org or project?)
- update pyamg.org
- specific points 
- point to code
- point to examples
- analytics (google or bitdeli?)
- update install scripts
- clearly block out strength, aggregation, interp routines
- move BImprove before strength?  (test before/after)
- clean up symmetry flag

- move default option from preprocess_Bimprove somewhere else, just set the
  default to ('gauss_seidel', {'sweep':'symmetric', 'iterations':4}), but then
  you have to worry about symmetry.
- change Bimprove to improve_candidates  (consistent with adaptive SA solver)
- change 'scheme' to consistent usage
- rename preprocess_Bimprove, preprocess_str_or_agg, preprocess_smooth to levelize
- fix returns for preprocess
- preprocess stuff needs full doc-strings
- As you change the preprocess names: also update test_utils, __all__ in utils and rootnode and smoothed_aggregation_solver

