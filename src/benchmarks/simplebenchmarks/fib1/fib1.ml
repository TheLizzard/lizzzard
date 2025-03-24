let rec fib n =
  if n < 1 then
    1
  else
    fib (n - 2) + fib (n - 1)

let () =
  let n = int_of_string Sys.argv.(1) in
  let result = fib n in
  Printf.printf "fib(%d)=%d\n" n result