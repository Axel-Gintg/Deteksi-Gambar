from google.colab import files as colab_files


def deteksi_gambar(threshold: float = 0.5) -> None:
    """
    Upload satu atau lebih gambar dan tampilkan hasil deteksi.

    Args:
        threshold: Ambang batas probabilitas untuk klasifikasi tampered (default: 0.5).
    """
    print("📁 Pilih gambar yang ingin dideteksi:")
    uploaded = colab_files.upload()

    for filename, content in uploaded.items():
        # Simpan file ke disk lokal
        tmp_file = Path(filename)
        tmp_file.write_bytes(content)

        # ─── Prediksi ────────────────────────────────────────────────────────
        model.eval()
        ela_pil = convert_to_ela(str(tmp_file))
        tensor  = val_tfm(image=np.array(ela_pil))["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out            = model(tensor)
            probs          = torch.softmax(out, 1)[0]
            prob_tampered  = probs[1].item()
            prob_authentic = probs[0].item()

        hasil = "TAMPERED 🔴" if prob_tampered >= threshold else "AUTHENTIC 🟢"

        # ─── Visualisasi ─────────────────────────────────────────────────────
        orig     = Image.open(str(tmp_file)).convert("RGB")
        ela_disp = convert_to_ela(str(tmp_file))

        # Grad-CAM (inisialisasi per-gambar agar thread-safe)
        cam_local = GradCAM(model=model, target_layers=[model.blocks[-1]])
        grayscale = cam_local(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(1)],
        )[0]
        orig_np  = np.array(orig.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
        cam_img  = show_cam_on_image(orig_np, grayscale, use_rgb=True)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        axes[0].imshow(orig)
        axes[0].set_title("Gambar Input", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(ela_disp)
        axes[1].set_title("ELA — Area terang = mencurigakan", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(cam_img)
        axes[2].set_title("Grad-CAM — Area yang dideteksi model", fontsize=11)
        axes[2].axis("off")

        warna = "red" if prob_tampered >= threshold else "green"
        plt.suptitle(
            f"Hasil: {hasil}\n"
            f"Confidence Tampered: {prob_tampered:.1%}  |  Authentic: {prob_authentic:.1%}",
            fontsize=13, color=warna, fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        print(f"\n{'='*45}")
        print(f"  File      : {filename}")
        print(f"  Hasil     : {hasil}")
        print(f"  Tampered  : {prob_tampered:.1%}")
        print(f"  Authentic : {prob_authentic:.1%}")
        print(f"{'='*45}\n")

        # Bersihkan file sementara
        tmp_file.unlink(missing_ok=True)


# ─── Jalankan deteksi ────────────────────────────────────────────────────────
deteksi_gambar()
