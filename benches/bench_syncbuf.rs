#![feature(generic_associated_types)]

use std::time::Duration;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, BatchSize};
use crossbeam::thread::scope;
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use syncbuf::Syncbuf;
use parking_lot::RwLock;

fn syncbuf_iterate_uncontested(buf: Syncbuf<usize>, threads: usize) {
    scope(|s| {
        for _ in 0..threads {
            s.spawn(|_| {
                for e in buf.iter() {
                    black_box(e);
                }
            });
        }
    }).expect("some thread failed");
}

fn lock_iterate_uncontested(buf: RwLock<Vec<usize>>, threads: usize) {
    scope(|s| {
        for _ in 0..threads {
            s.spawn(|_| {
                for e in buf.read().iter() {
                    black_box(e);
                }
            });
        }
    }).expect("some thread failed");
}

fn syncbuf_iterate_contested(buf: Syncbuf<usize>, num: usize, readers: usize, writers: usize) {
    let chunk_size = num / writers;
    scope(|s| {
        for _ in 0..writers {
            s.spawn(|_| {
                for e in 0..chunk_size {
                    buf.push(e).expect("overflow");
                }
            });
        }
        for _ in 0..readers {
            s.spawn(|_| {
                for e in buf.iter() {
                    black_box(e);
                }
            });
        }
    }).expect("some thread failed");
}

fn lock_iterate_contested(buf: RwLock<Vec<usize>>, num: usize, readers: usize, writers: usize) {
    let chunk_size = num / writers;
    let buf_ref = &buf;
    scope(|s| {
        for _ in 0..writers {
            s.spawn(move |_| {
                for e in 0..chunk_size {
                    buf_ref.write().push(e);
                }
            });
        }
        for _ in 0..readers {
            s.spawn(|_| {
                for e in buf_ref.read().iter() {
                    black_box(e);
                }
            });
        }
    }).expect("some thread failed");
}

fn bench_syncbuf(c: &mut Criterion) {
    let mut group = c.benchmark_group("Iterate Uncontested");
    let start_size = 512_000usize;
    for i in [start_size, start_size*32, /* start_size*32*32 */].iter() {
        let mut rand_vec = vec![0usize; *i];
        thread_rng().fill(&mut*rand_vec);

        group.bench_with_input(
            BenchmarkId::new("Syncbuf", rand_vec.len()),
            &rand_vec,
            |b, input| {
                b.iter_batched(
                    || Syncbuf::from(input.clone()),
                    |i| syncbuf_iterate_uncontested(i, 16),
                    BatchSize::LargeInput)
            });
        group.bench_with_input(
            BenchmarkId::new("RwLock", rand_vec.len()),
            &rand_vec,
            |b, input| {
                b.iter_batched(
                    || RwLock::new(input.clone()),
                    |i| lock_iterate_uncontested(i, 16),
                    BatchSize::LargeInput)
            });

    }
    group.finish();

    let mut group = c.benchmark_group("Iterate With Single Writer");
    let start_size = 512_000usize;
    for i in [start_size, start_size*4, start_size*16].iter() {
        let mut rand_vec = Vec::with_capacity(*i);
        let num_unwritten = i / 4;
        for rand in thread_rng().sample_iter(Uniform::new(0, usize::MAX)).take(i - num_unwritten) {
            rand_vec.push(rand);
        }

        group.bench_with_input(
            BenchmarkId::new("Syncbuf", i),
            &rand_vec,
            |b, input| {
                b.iter_batched(
                    || {
                        let mut new_input = input.clone();
                        new_input.reserve(num_unwritten);
                        Syncbuf::from(new_input)
                    },
                    |i| syncbuf_iterate_contested(i, num_unwritten, 16, 1),
                    BatchSize::LargeInput)
            });
        group.bench_with_input(
            BenchmarkId::new("RwLock", i),
            &rand_vec,
            |b, input| {
                b.iter_batched(
                    || RwLock::new(input.clone()),
                    |i| lock_iterate_contested(i, num_unwritten, 16, 1),
                    BatchSize::LargeInput)
            });
    }
    group.finish();

    let mut group = c.benchmark_group("Iterate With Contention");
    let start_size = 512_000usize;
    for i in [start_size, start_size*4, start_size*16].iter() {
        let mut rand_vec = Vec::with_capacity(*i);
        let num_unwritten = i / 4;
        for rand in thread_rng().sample_iter(Uniform::new(0, usize::MAX)).take(i - num_unwritten) {
            rand_vec.push(rand);
        }

        group.bench_with_input(
            BenchmarkId::new("Syncbuf", i),
            &rand_vec,
            |b, input| {
                b.iter_batched(
                    || {
                        let mut new_input = input.clone();
                        new_input.reserve(num_unwritten);
                        Syncbuf::from(new_input)
                    },
                    |i| syncbuf_iterate_contested(i, num_unwritten, 16, 16),
                    BatchSize::LargeInput)
            });
        group.bench_with_input(
            BenchmarkId::new("RwLock", i),
            &rand_vec,
            |b, input| {
                b.iter_batched(
                    || RwLock::new(input.clone()),
                    |i| lock_iterate_contested(i, num_unwritten, 16, 16),
                    BatchSize::LargeInput)
            });
    }
    group.finish();
}

criterion_group!{
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::new(1, 0))
        .measurement_time(Duration::new(3, 0))
        .noise_threshold(0.05)
        .sample_size(50);
    targets = bench_syncbuf
}
criterion_main!(benches);
