let () =
  let n = int_of_string Sys.argv.(1) in
  let i = ref 0 in
  while !i < n do
    i := !i + 1
  done;
  Printf.printf "%d\n" !i