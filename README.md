# structure-property
Particle based structural characterization.
本文件用于计算颗粒体系的结构序特征。

输入：直接由LIGGGHTS输出的颗粒位置文件。(dump-   .sample)

输出：包含三个sheet的xlsx表格文件。
      sheet1：symmetry feature 72个特征；
      sheet2：interstice distribution 60个特征；
      sheet3：conventional feature 265个特征；
            sheet3包含：1、Coordination number by Voronoi tessellation、Coordination number by cutoff distance (SRO and MRO).
                        Reference[1]: Okabe, A., Boots, B., Sugihara, K. & Chiu, S. N. Spatial Tesselations. Concepts and Applications of                           Voronoi Diagrams (John Wiley & Sons, 2009).
                        2、Voronoi idx3…7(SRO and MEO).
                        Reference[1]: Okabe, A., Boots, B., Sugihara, K. & Chiu, S. N. Spatial Tesselations. Concepts and Applications of                           Voronoi Diagrams (John Wiley & Sons, 2009).
                        3、Local volume fraction(SRO and MEO).
                        4、i-fold symm idx3...7(SRO and MRO).
                        Reference[2]: Peng, H. L., Li, M. Z. & Wang, W. H. Structural signature of plastic deformation in metallic glasses.                         Phys. Rev. Lett. 106, 135503 (2011).
                        5、weighted i-fold symm idx3…7(SRO and MRO).
                        6、Bond orientation order parameters(SRO and MRO).
                        7、Modified BOO(SRO and MRO).
                        Reference[3]:https://pyboo.readthedocs.io/en/latest/intro.html
                        Reference[4]:Xia, C. et al. The structural origin of the hard-sphere glass transition in granular packing. Nat.                             Commun.
                        Reference[5]: Clusters of polyhedra in spherical confinement. Proc. Natl. Acad. Sci. U. S. A.
                        8、Cluster packing efficiency(SRO and MRO).
                        Reference[6]: Yang, L. et al. Atomic-scale mechanisms of the glass-forming ability in metallic glasses. Phys.                           Rev. Lett. 109, 105502 (2012).
