from psdr import BoxDomain, voronoi_vertex

dom = BoxDomain([0,0], [1,1])
Xhat = dom.sample(10)
X0 = dom.sample(5)

voronoi_vertex(dom, Xhat, X0)

