let rec f x =
  if x > 0 then
    f (x - 1) + 1
  else
    0

let () =
  let n = int_of_string Sys.argv.(1) in
  Printf.printf "%d\n" (f n)

(* StackOverflow *)