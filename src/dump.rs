

pub fn dump(name: &str, grbuf: &[f32]) {
    print!("{name}");
    for x in grbuf {
        print!(" {x}");
    }
    println!()
}
