# 🧠 Memvid - Memori Portabel untuk AI Agent

Memvid adalah lapisan memori berkas tunggal untuk agen AI dengan pengambilan instan dan memori jangka panjang. Memori persisten, berversi, dan portabel, tanpa database.

## Ikhtisar

Memvid menyatukan data, embedding, struktur pencarian, dan metadata Anda ke dalam satu berkas portabel (.mv2). Tidak perlu pipeline RAG kompleks atau database vektor server. Cukup satu berkas—cepat diakses, mudah dibawa.

## Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| **Living Memory Engine** | Tambah, cabang, dan evolusikan memori lintas sesi secara berkelanjutan |
| **Kapsul Konteks (.mv2)** | Kapsul memori mandiri yang dapat dibagikan dengan aturan dan masa berlaku |
| **Time-Travel Debugging** | Mundur, putar ulang, atau cabang status memori kapan saja |
| **Smart Recall** | Akses memori lokal di bawah 5ms dengan caching prediktif |
| **Codec Intelligence** | Kompresi yang dipilih dan ditingkatkan secara otomatis seiring waktu |

## Benchmark

- **Akurasi lebih tinggi:** +35% SOTA pada LoCoMo
- **Penalaran multi-hop & temporal:** +76% multi-hop, +56% temporal
- **Latensi ultra-rendah:** 0.025ms P50, 0.075ms P99
- **Throughput 1.372× lebih tinggi** dibanding standar

## Mulai Cepat

```bash
cargo add memvid-core
```

Atau instal CLI:

```bash
cargo install memvid-cli
```

## Penggunaan Dasar

```rust
use memvid_core::Memvid;

// Buat atau buka memori
let mut mem = Memvid::create("memory.mv2")?;
mem.put_text("Halo, dunia!")?;
mem.commit()?;

// Cari
let results = mem.search("halo")?;
for r in results {
    println!("Skor: {} - {}", r.score, r.content);
}
```

## Arsitektur

Berkas .mv2 terdiri dari: Header (4KB) → WAL Tertanam → Segmen Data → Indeks Teks → Indeks Vektor → Indeks Waktu → Daftar Isi

Semua operasi melalui WAL (Write-Ahead Log) untuk keamanan crash. Bingkai bersifat append-only dan tidak dapat diubah setelah dikomit.

## Sumber Daya

- [Dokumentasi](https://docs.memvid.com)
- [Coba Sandbox](https://sandbox.memvid.com)
- [Diskusi](https://github.com/memvid/memvid/discussions)
- [Laporan Bug](https://github.com/memvid/memvid/issues)

## Lisensi

Apache 2.0
