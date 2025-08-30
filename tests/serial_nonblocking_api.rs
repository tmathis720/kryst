use kryst::parallel::{Comm, NoComm, UniverseComm};

#[test]
fn serial_nonblocking_request_lifecycle() {
    // Use serial backend and exercise irecv/isend/wait_all with the shared Request type
    let comm = UniverseComm::NoComm(NoComm);

    let mut recv = vec![0.0f64; 4];
    let send = vec![1.0f64; 4];

    let mut reqs: Vec<<UniverseComm as Comm>::Request<'_>> = Vec::new();
    reqs.push(comm.irecv_from(&mut recv[..], 0));
    reqs.push(comm.isend_to(&send[..], 0));
    comm.wait_all(&mut reqs);
    // Ensure requests are dropped before reading recv buffer
    drop(reqs);

    // In serial, no data moves; the important part is type/lifetime integration.
    assert_eq!(recv, vec![0.0; 4]);
}
