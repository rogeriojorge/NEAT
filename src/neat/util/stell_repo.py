from qsc import Qsc


def get_stel(nphi, ind):
    if ind == 0:
        # Tokamak case
        eR = [1.0]
        eZ = [0]
        eta = -0.7
        B2c = -3.2
        I2 = 2
        B0 = 3
        stel = Qsc(
            rc=eR,
            zs=eZ,
            etabar=eta,
            nfp=2,
            nphi=nphi,
            I2=I2,
            order="r2",
            B0=B0,
            B2c=B2c,
        )
        name = "tokamak"  # name of results folder and files
        r0 = 0.03
        # Lambda = [0.35,-0.35] # trapped particle case
        # Lambda = [0.33,-0.33] # trapped particle case
        # Lambda = [0.2,-0.2] # passing particle case
        Lambda = [0.32]  # passing particle case
    if ind == 1:
        paper_case = 1
        stel = Qsc.from_paper(paper_case, nphi=nphi, B0=3)
        name = "paper_r2_5." + str(paper_case)  # name of results folder and files
        r0 = 0.1
        Lambda = [0.349, -0.349]
    if ind == 2:
        paper_case = 2
        stel = Qsc.from_paper(paper_case, nphi=nphi, B0=3)
        name = "paper_r2_5." + str(paper_case)  # name of results folder and files
        # r0       = 0.1
        r0 = 0.005
        # Lambda   = [0.349,-0.349]
        Lambda = [0.2]
    if ind == 3:
        paper_case = 3
        stel = Qsc.from_paper(paper_case, nphi=nphi, B0=3)
        name = "paper_r2_5." + str(paper_case)  # name of results folder and files
        r0 = 0.1
        Lambda = [0.36, -0.36]
    if ind == 4:
        paper_case = 4
        stel = Qsc.from_paper(paper_case, nphi=nphi, B0=3)
        name = "paper_r2_5." + str(paper_case)  # name of results folder and files
        r0 = 0.01
        # Lambda   = [0.385,-0.385]
        Lambda = [0.2]
    if ind == 5:
        stel = Qsc(
            rc=[1, 0.17, 0.01804, 0.001409, 5.877e-05],
            zs=[0, 0.1581, 0.01820, 0.001548, 7.772e-05],
            nfp=4,
            etabar=1.569,
            order="r2",
            B2c=-2.5,
            B0=2,
        )
        name = "paper_r2_5.4_modified"  # name of results folder and files
        r0 = 0.1
        Lambda = [0.42, -0.42]
    if ind == 6:
        paper_case = "r1 section 5.2"
        stel = Qsc.from_paper(paper_case, nphi=nphi, B0=3)
        name = "paper_r1_5.2"  # name of results folder and files
        r0 = 0.1
        Lambda = [0.2993]
    return stel, name, r0, Lambda
