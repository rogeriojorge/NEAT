module neatf90
    use neo_orb, only: NeoOrb

    implicit none

    type(NeoOrb) :: norb
end module neatf90

subroutine init(vmec_file, ns_s, ns_tp, multharm, integmode)
    use neo_orb, only: init_field
    use neatf90, only: norb

    implicit none

    character(len=*), intent(in) :: vmec_file
    integer, intent(in) :: ns_s, ns_tp, multharm, integmode
    
    call init_field(norb, vmec_file, ns_s, ns_tp, multharm, integmode)

    ! TODO: 
    ! init_params
    ! pre-compute starting flux surface
    ! npoi=nper*npoiper ! total number of starting points
    ! allocate(xstart(3,npoi),bstart(npoi),volstart(npoi))
    ! call init_starting_surf
end subroutine init

subroutine init_orbit(z_vmec, z_can, dtau, relerr)
    use neo_orb, only: init_sympl
    use neatf90, only: norb

    implicit none

    double precision, intent(in)  :: z_vmec(3)  ! Initial condition in VMEC coordinates
    double precision, intent(out) :: z_can(3)   ! Initial condition in canonical flux coordinates
    double precision :: dtau    ! Timestep
    double precision, optional :: relerr  ! Relative error

    IF(.not. present(relerr)) relerr = 1d-13
    
    z_can(1)=z_vmec(1)
    call vmec_to_can(z_vmec(1),z_vmec(2),z_vmec(3),z_can(2),z_can(3))
    
    call init_sympl(norb%si, norb%f, z_can, dtau, dtau, relerr, norb%integmode)
end subroutine init_orbit

subroutine timestep()
end subroutine timestep
