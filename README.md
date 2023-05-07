

Input from CPU side: OVM Mesh
    vertices: pos
    tets: vidx[4]

Compute pipeline CellScaler:
    input: mesh
    uniform: scale factor
    output "tet vertex positions":
        tets: pos[4]

Compute pipeline ViewPrecompute:
    uniform: camera position in object coordinate space
    input: tet vertex positions
    output:





tet-ray intersection
- could optimize for front face (need to render triangle insteances instead of tet instances, so the triangle instance tells us what the front face is)
- for back face: always need to solve system

triangles
    inner faces:
        abc, bcd, adb, acd
        "a" for triangle "abc" in text below relies to the first vertex
        baca: b-a cross c-a  \
        cbdb                  | view independent
        daba                  |
        cada                 /

        precompute xyzy dot o-a?!



Tet a, b, c, d
ray o+tg (g as ray direction)
    o, g already in object coordinate system

    triangle abc
    g = pos - o

    a + [[b - a] [c - a]] * [u v] = o + tg
    [[b - a] [c - a] [g]] * [u v t] = o-a
    \-------------------/
             = A

    cramer's rule

         det [[b - a] [c - a] [o-a]]    (fixed for fixed o)
    t =  ---------------------------
                  det A                 (depends on position)

         det [[o -a ] [c - a] [g]]      (depends on position)
    u =  ---------------------------
                  det A                 (depends on position)
         cross([o -a ] [c - a]) dot g      (depends on position) (check sign)
      =  ---------------------------
                  det A                 (depends on position)


    Q: do we need to compute uv, for [0,1] bounds check, or is the lowest t always in bounds?
        i think checking the sign of (non-normalized) normal vs view direction suffices! see if we hit a triangle "from the inside";
precompute per triangle:
   nominator for t computation: det [[b - a] [c - a] [o-a]]
   [b-a] cross [c-a] for fast determinants




<x, a cross b> = det [x a b]
