import argparse
import sys
import numpy as np
import tensorflow as tf

@tf.function
def rotate(points, theta):
    rotation_matrix = tf.stack(
        [tf.cos(theta), -tf.sin(theta), tf.sin(theta), tf.cos(theta)]
    )
    rotation_matrix = tf.reshape(rotation_matrix, (2, 2))
    return tf.matmul(rotation_matrix, points)


def solve_linear_system(A, b):
    det = tf.linalg.det(A)
    if tf.math.abs(det) < 1e-8:
        raise ValueError(
            "Układ nie ma rozwiązań lub ma nieskończenie wiele (det(A)≈0)."
        )
    x = tf.linalg.solve(A, b)
    return x

def main(argv=None):

    parser = argparse.ArgumentParser(description="Lab 2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- rotate ---
    p_rot = sub.add_parser("rotate", help="Obrót punktuow wokol (0 . 0).")
    p_rot.add_argument("--degree", type=float, required=True, help="Kąt w stopniach.")
    p_rot.add_argument("x", type=float, help="x")
    p_rot.add_argument("y", type=float, help="y")

    # --- solve ---
    p_sol = sub.add_parser("solve", help="Rozwiąż układ A x = b.")
    p_sol.add_argument("-n", type=int, required=True)
    p_sol.add_argument("-A", type=float, nargs="+", required=True)
    p_sol.add_argument("-b", type=float, nargs="+", required=True)

    args = parser.parse_args(argv)

    # python .\LAB2\main.py rotate --degree 30 2 0
    if args.cmd == "rotate":
        theta_rad = np.deg2rad(np.float32(args.degree))
        point = tf.constant([[args.x], [args.y]], dtype=tf.float32)  
        out = rotate(point, tf.constant(theta_rad, dtype=tf.float32))
        out = tf.round(out * 1e6) / 1e6  # round to 6 decimal places
        print(out.numpy(), flush=True)
        return 0

    # python .\LAB2\main.py solve -n 3 -A 3 2 -2 1 1 1 2 2 1 -b 3 3 5
    if args.cmd == "solve":
        A = tf.constant(np.array(args.A, dtype=np.float32).reshape(args.n, args.n))
        b = tf.constant(np.array(args.b, dtype=np.float32).reshape(args.n, 1))
        x = solve_linear_system(A, b)
        x = tf.round(x * 1e6) / 1e6  # round to 6 decimal places
        print(x.numpy().flatten(), flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
    