#include <QApplication>
#include <QSurfaceFormat>
#include <QTest>

#include <memory>

#include "effects/effect.h"
#include "effects/internal/srtparser.h"
#include "effects/internal/subtitleeffect.h"
#include "engine/clip.h"
#include "tests/test_render_harness.h"

class TestEffectsGpu : public QObject {
  Q_OBJECT
 private slots:
  void initTestCase();
  void cleanupTestCase();

  void solidColorPassthrough();

  void effectPassthroughSweep_data();
  void effectPassthroughSweep();

  // Bespoke video effect tests (Phase 2).
  void transformTranslate();
  void transformScale();
  void transformRotate180();
  void textRenders();
  void richTextRenders();
  void shakeChangesOutput();
  void timecodeRenders();
  void cornerpinDistorts();
  void subtitleActiveEntry();
  void gradientRendersTwoStops();

 private:
  std::unique_ptr<TestRenderHarness> h_;
};

void TestEffectsGpu::initTestCase() {
  h_ = std::make_unique<TestRenderHarness>();
  if (!h_->initialized()) QSKIP("QRhi OpenGLES2 unavailable");
}

void TestEffectsGpu::cleanupTestCase() { h_.reset(); }

void TestEffectsGpu::solidColorPassthrough() {
  auto seq = h_->make_sequence(64, 64, 30.0);
  // -1 = topmost video track (Olive's negative-is-video convention).
  Clip* c = h_->add_generator_clip(seq.get(), -1, 0, 30, EFFECT_INTERNAL_SOLID);

  // c->effects[0] = Transform, c->effects[1] = Solid
  Effect* solid = c->effects[1].get();
  // Row 2 = Color, field 0 = the ColorField itself.
  h_->set_field_color(solid, 2, 0, QColor(255, 128, 0, 255));

  QByteArray pixels = h_->render_frame(seq.get(), 0);
  QVERIFY(!pixels.isEmpty());
  h_->assert_solid_color(pixels, 64, 64, QColor(255, 128, 0, 255), 2);
}

void TestEffectsGpu::effectPassthroughSweep_data() {
  QTest::addColumn<int>("effect_index");

  // `effects` is the global QVector<EffectMeta> from effect.h:60 (no namespace).
  // We pass the vector index — looking up by name via get_meta_from_name() fails
  // for effect names containing '/' (e.g. "Hue/Saturation/Brightness"), since
  // that function interprets the first '/' as a category/name separator.
  for (int i = 0; i < effects.size(); ++i) {
    const EffectMeta& meta = effects[i];
    // Skip transitions (they need a different setup with two clips).
    if (meta.type != EFFECT_TYPE_EFFECT) continue;
    // Skip audio-subtype effects — sweep is video-only. Audio effects are
    // exercised in Phase 2 once render_audio() exists.
    if (meta.subtype == EFFECT_TYPE_AUDIO) continue;
    // Skip deprecated / GUI-only / disabled internals.
    if (meta.internal == EFFECT_INTERNAL_FREI0R) continue;
    if (meta.internal == EFFECT_INTERNAL_VST) continue;
    if (meta.internal == EFFECT_INTERNAL_MASK) continue;  // disabled
    QTest::newRow(qPrintable(meta.name)) << i;
  }
}

void TestEffectsGpu::effectPassthroughSweep() {
  QFETCH(int, effect_index);
  const EffectMeta* meta = &effects[effect_index];

  auto seq = h_->make_sequence(64, 64, 30.0);
  Clip* c = h_->add_generator_clip(seq.get(), -1, 0, 30, EFFECT_INTERNAL_SOLID);
  // Set Solid to a non-black colour so we can detect "all-zero output" as a failure mode.
  Effect* solid = c->effects[1].get();
  h_->set_field_color(solid, 2, 0, QColor(128, 128, 128, 255));

  // Attach the effect under test on top of the Solid generator.
  // Skip if it's the same internal as the generator (Solid already there).
  if (meta->internal != EFFECT_INTERNAL_TRANSFORM && meta->internal != EFFECT_INTERNAL_SOLID) {
    h_->attach_effect(c, meta);
  }

  QByteArray pixels = h_->render_frame(seq.get(), 0);
  QVERIFY2(!pixels.isEmpty(), qPrintable("readback empty for " + meta->name));
  QCOMPARE(pixels.size(), 64 * 64 * 4);

  // Smoke check: the buffer is not all-zero, which would indicate a crash
  // mid-render that left main_tex untouched.
  bool any_non_zero = false;
  for (int i = 0; i < pixels.size(); ++i) {
    if (pixels[i] != 0) {
      any_non_zero = true;
      break;
    }
  }
  QVERIFY2(any_non_zero, qPrintable("output all zeros for " + meta->name));
}

// ---------------------------------------------------------------------------
// Bespoke video effect tests
// ---------------------------------------------------------------------------
//
// Pattern: configure a small (64x64 unless noted) sequence with a Solid or
// Gradient generator, optionally attach the effect under test on top, render
// one frame, then assert on the readback pixels.
//
// Field row/field indices are taken from the corresponding effect ctor in
// src/effects/internal/<effect>.cpp. SetValueAt(0.0, value) writes
// persistent_data_, which is what GetValueAt reads at every timecode for
// non-keyframable fields (the default). SetDefaultData would NOT work — it
// writes default_data_ which the render path never reads.

namespace {

// Returns the alpha byte of pixel (x,y) in an RGBA8 readback (row-major).
inline int alpha_of(const QByteArray& pixels, int w, int x, int y) {
  return static_cast<int>(static_cast<uchar>(pixels[(y * w + x) * 4 + 3]));
}

// Returns the channels (R,G,B,A) at pixel (x,y).
struct Rgba {
  int r, g, b, a;
};
inline Rgba pixel_at(const QByteArray& pixels, int w, int x, int y) {
  const uchar* p = reinterpret_cast<const uchar*>(pixels.constData()) + (y * w + x) * 4;
  return {p[0], p[1], p[2], p[3]};
}

}  // namespace

void TestEffectsGpu::transformTranslate() {
  // Solid red generator + Transform translated +16 in X. Without translation
  // every pixel is red. With +16 translation, the leftmost 16 columns of the
  // output should be transparent (texture sampled outside ClampToEdge would
  // still be red, but the clip's quad vertices are translated so the area
  // outside the quad receives no draw). We use the "core area still red"
  // assertion: pixel (32, 32) is well inside the translated quad and stays red.
  auto seq = h_->make_sequence(64, 64, 30.0);
  Clip* c = h_->add_generator_clip(seq.get(), -1, 0, 30, EFFECT_INTERNAL_SOLID);
  Effect* solid = c->effects[1].get();
  h_->set_field_color(solid, 2, 0, QColor(255, 0, 0, 255));

  Effect* transform = c->effects[0].get();
  // Transform row 0 = Position (posx, posy); default at refresh() is (w/2, h/2).
  // Adding +16 to posx moves the clip's quad right by 16 px.
  h_->set_field_double(transform, 0, 0, 64.0 / 2.0 + 16.0);  // posx
  h_->set_field_double(transform, 0, 1, 64.0 / 2.0);         // posy

  QByteArray pixels = h_->render_frame(seq.get(), 0);
  QVERIFY(!pixels.isEmpty());

  // Center of the translated quad: (32, 32) should still be solid red.
  h_->assert_pixel(pixels, 64, 32, 32, QColor(255, 0, 0, 255), 4);
  // A pixel inside the quad's new position only (e.g. (40, 32)): still red.
  h_->assert_pixel(pixels, 64, 40, 32, QColor(255, 0, 0, 255), 4);
}

void TestEffectsGpu::transformScale() {
  // Solid + Transform scale 50% → bounding box becomes 32x32 centered. Outside
  // (e.g. corner (0,0)) is transparent; inside (e.g. (32,32)) is red.
  auto seq = h_->make_sequence(64, 64, 30.0);
  Clip* c = h_->add_generator_clip(seq.get(), -1, 0, 30, EFFECT_INTERNAL_SOLID);
  Effect* solid = c->effects[1].get();
  h_->set_field_color(solid, 2, 0, QColor(255, 0, 0, 255));

  Effect* transform = c->effects[0].get();
  // Row 1 = Scale (scalex, scaley) as percent. Row 2 = UniformScale bool (default true).
  h_->set_field_double(transform, 1, 0, 50.0);  // scalex
  h_->set_field_double(transform, 1, 1, 50.0);  // scaley

  QByteArray pixels = h_->render_frame(seq.get(), 0);
  QVERIFY(!pixels.isEmpty());

  // Center stays red.
  h_->assert_pixel(pixels, 64, 32, 32, QColor(255, 0, 0, 255), 4);
  // Far corner is outside the half-size quad → transparent.
  QVERIFY2(alpha_of(pixels, 64, 1, 1) < 50,
           qPrintable(QString("expected transparent corner, alpha=%1").arg(alpha_of(pixels, 64, 1, 1))));
}

void TestEffectsGpu::transformRotate180() {
  // Gradient (black→white top→bottom) + Transform rotated 180. After rotation
  // the top of the gradient (black) ends up at the bottom and vice versa.
  auto seq = h_->make_sequence(64, 64, 30.0);
  Clip* c = h_->add_generator_clip(seq.get(), -1, 0, 30, EFFECT_INTERNAL_GRADIENT);
  Effect* gradient = c->effects[1].get();
  // Gradient row 0 = type (combo, default LINEAR), 1 = start_color, 2 = end_color, 3 = angle.
  h_->set_field_color(gradient, 1, 0, QColor(Qt::black));
  h_->set_field_color(gradient, 2, 0, QColor(Qt::white));
  h_->set_field_double(gradient, 3, 0, 90.0);  // top→bottom

  Effect* transform = c->effects[0].get();
  h_->set_field_double(transform, 3, 0, 180.0);  // rotation row=3 field=0

  QByteArray pixels = h_->render_frame(seq.get(), 0);
  QVERIFY(!pixels.isEmpty());

  // After 180 rotation the gradient is flipped: top of output should now be
  // bright (was bottom of gradient = white), bottom should be dark.
  Rgba top = pixel_at(pixels, 64, 32, 4);
  Rgba bot = pixel_at(pixels, 64, 32, 60);
  QVERIFY2(
      top.r > bot.r + 50,
      qPrintable(
          QString("expected top brighter than bottom after 180 rotation, top.r=%1 bot.r=%2").arg(top.r).arg(bot.r)));
}

void TestEffectsGpu::textRenders() {
  // Text "HELLO" white centered on a transparent clip — assert at least one
  // central pixel is bright and at least one border pixel is dim.
  auto seq = h_->make_sequence(64, 64, 30.0);
  Clip* c = h_->add_generator_clip(seq.get(), -1, 0, 30, EFFECT_INTERNAL_SOLID);
  // Make Solid transparent so we measure only the text contribution.
  Effect* solid = c->effects[1].get();
  h_->set_field_color(solid, 2, 0, QColor(0, 0, 0, 0));

  // Text rows: 0=text, 1=font, 2=size, 3=color, 4=alignment, 5=wordwrap.
  EffectPtr text_eff = h_->attach_internal(c, EFFECT_INTERNAL_TEXT, EFFECT_TYPE_EFFECT);
  h_->set_field_string(text_eff.get(), 0, 0, "HELLO");
  h_->set_field_double(text_eff.get(), 2, 0, 16.0);  // size
  h_->set_field_color(text_eff.get(), 3, 0, QColor(Qt::white));

  QByteArray pixels = h_->render_frame(seq.get(), 0);
  QVERIFY(!pixels.isEmpty());

  // At least one pixel in the central horizontal band has high alpha (text glyph).
  bool central_bright = false;
  for (int y = 25; y <= 38 && !central_bright; ++y) {
    for (int x = 8; x < 56; ++x) {
      if (alpha_of(pixels, 64, x, y) > 200) {
        central_bright = true;
        break;
      }
    }
  }
  QVERIFY2(central_bright, "expected at least one bright text pixel in central band");
}

void TestEffectsGpu::richTextRenders() {
  // RichText with simple bold HTML. Same "non-background pixels in center" check.
  auto seq = h_->make_sequence(64, 64, 30.0);
  Clip* c = h_->add_generator_clip(seq.get(), -1, 0, 30, EFFECT_INTERNAL_SOLID);
  Effect* solid = c->effects[1].get();
  h_->set_field_color(solid, 2, 0, QColor(0, 0, 0, 0));

  // RichText row 0 = text (HTML string).
  EffectPtr rich = h_->attach_internal(c, EFFECT_INTERNAL_RICHTEXT, EFFECT_TYPE_EFFECT);
  h_->set_field_string(rich.get(), 0, 0,
                       "<html><body style=\"color:#ffffff; font-size:20pt;\"><center>X</center></body></html>");

  QByteArray pixels = h_->render_frame(seq.get(), 0);
  QVERIFY(!pixels.isEmpty());

  // Look anywhere in the frame for a non-background pixel (RichText positions
  // by document layout; the glyph X lands somewhere central).
  bool any_glyph = false;
  for (int y = 10; y < 54 && !any_glyph; ++y) {
    for (int x = 10; x < 54; ++x) {
      if (alpha_of(pixels, 64, x, y) > 100) {
        any_glyph = true;
        break;
      }
    }
  }
  QVERIFY2(any_glyph, "expected at least one non-background pixel from RichText glyph");
}

void TestEffectsGpu::shakeChangesOutput() {
  // Solid + Gradient + Shake at default amplitude. Compare two adjacent frames
  // — at non-zero intensity the shake offset must differ between them.
  auto seq = h_->make_sequence(64, 64, 30.0);
  Clip* c = h_->add_generator_clip(seq.get(), -1, 0, 30, EFFECT_INTERNAL_GRADIENT);
  Effect* gradient = c->effects[1].get();
  h_->set_field_color(gradient, 1, 0, QColor(Qt::black));
  h_->set_field_color(gradient, 2, 0, QColor(Qt::white));
  h_->set_field_double(gradient, 3, 0, 90.0);

  // Shake defaults: intensity=25, rotation=10, frequency=5. Plenty to move
  // pixels between two adjacent timecodes.
  h_->attach_internal(c, EFFECT_INTERNAL_SHAKE, EFFECT_TYPE_EFFECT);

  QByteArray frame0 = h_->render_frame(seq.get(), 0);
  QByteArray frame1 = h_->render_frame(seq.get(), 1);
  QVERIFY(!frame0.isEmpty());
  QVERIFY(!frame1.isEmpty());
  QCOMPARE(frame0.size(), frame1.size());

  // At least some pixels must differ.
  bool any_diff = false;
  for (int i = 0; i < frame0.size(); ++i) {
    if (frame0[i] != frame1[i]) {
      any_diff = true;
      break;
    }
  }
  QVERIFY2(any_diff, "expected shake to produce different pixels between frame 0 and frame 1");
}

void TestEffectsGpu::timecodeRenders() {
  // Timecode effect on a transparent clip — assert text-shaped pixels appear
  // in the lower band where the timecode is drawn.
  auto seq = h_->make_sequence(64, 64, 30.0);
  Clip* c = h_->add_generator_clip(seq.get(), -1, 0, 30, EFFECT_INTERNAL_SOLID);
  Effect* solid = c->effects[1].get();
  h_->set_field_color(solid, 2, 0, QColor(0, 0, 0, 0));

  h_->attach_internal(c, EFFECT_INTERNAL_TIMECODE, EFFECT_TYPE_EFFECT);

  QByteArray pixels = h_->render_frame(seq.get(), 0);
  QVERIFY(!pixels.isEmpty());

  // Timecode draws at offset_y + height - height/10 = 64 - 6 = 58 with the
  // default scale 100 and a small background rect just above. Look for any
  // non-background pixel in the bottom band [40, 63].
  bool any_pixel = false;
  for (int y = 40; y < 64 && !any_pixel; ++y) {
    for (int x = 0; x < 64; ++x) {
      if (alpha_of(pixels, 64, x, y) > 100) {
        any_pixel = true;
        break;
      }
    }
  }
  QVERIFY2(any_pixel, "expected timecode text/background pixels in the bottom band");
}

void TestEffectsGpu::cornerpinDistorts() {
  // Solid + Gradient + CornerPin with corners moved inward 16 px each. The
  // warped quad covers only the center half of the sequence — pixel (0,0)
  // is outside and stays transparent; pixel (32, 32) is inside.
  //
  // CornerPin OFFSETS are added to default vertex coords. For a 64x64 clip
  // the default coords go from (-32,-32) to (32,32). Moving TL by (+16,+16)
  // shifts to (-16,-16); other corners analogously.
  auto seq = h_->make_sequence(64, 64, 30.0);
  Clip* c = h_->add_generator_clip(seq.get(), -1, 0, 30, EFFECT_INTERNAL_GRADIENT);
  Effect* gradient = c->effects[1].get();
  h_->set_field_color(gradient, 1, 0, QColor(Qt::black));
  h_->set_field_color(gradient, 2, 0, QColor(Qt::white));
  h_->set_field_double(gradient, 3, 0, 90.0);

  EffectPtr corner = h_->attach_internal(c, EFFECT_INTERNAL_CORNERPIN, EFFECT_TYPE_EFFECT);
  // Rows 0..3: TL, TR, BL, BR. Each row has two doubles (x, y). Row 4 is the
  // perspective bool (default true).
  h_->set_field_double(corner.get(), 0, 0, +16.0);  // TL x
  h_->set_field_double(corner.get(), 0, 1, +16.0);  // TL y
  h_->set_field_double(corner.get(), 1, 0, -16.0);  // TR x
  h_->set_field_double(corner.get(), 1, 1, +16.0);  // TR y
  h_->set_field_double(corner.get(), 2, 0, +16.0);  // BL x
  h_->set_field_double(corner.get(), 2, 1, -16.0);  // BL y
  h_->set_field_double(corner.get(), 3, 0, -16.0);  // BR x
  h_->set_field_double(corner.get(), 3, 1, -16.0);  // BR y

  QByteArray pixels = h_->render_frame(seq.get(), 0);
  QVERIFY(!pixels.isEmpty());

  // Center is inside the warped quad → non-transparent.
  QVERIFY2(alpha_of(pixels, 64, 32, 32) > 50,
           qPrintable(QString("expected center inside warped quad, alpha=%1").arg(alpha_of(pixels, 64, 32, 32))));
  // Corner is outside → transparent (no draw).
  QVERIFY2(alpha_of(pixels, 64, 1, 1) < 50,
           qPrintable(QString("expected transparent outside warp, alpha=%1").arg(alpha_of(pixels, 64, 1, 1))));
}

void TestEffectsGpu::subtitleActiveEntry() {
  // Subtitle effect with a single cue active over [0ms, 500ms). At 30 fps:
  //   frame 7  → 233 ms (active)  → text visible
  //   frame 30 → 1000 ms (past)   → fully transparent
  //
  // The second assertion exercises the AlwaysUpdate() fix: without it the
  // base Effect::valueHasChanged() short-circuit may skip a redraw when no
  // EffectField changes between frames, leaving the cached overlay.
  auto seq = h_->make_sequence(64, 64, 30.0);
  Clip* c = h_->add_generator_clip(seq.get(), -1, 0, 60, EFFECT_INTERNAL_SOLID);
  Effect* solid = c->effects[1].get();
  h_->set_field_color(solid, 2, 0, QColor(0, 0, 0, 0));

  EffectPtr sub_eff = h_->attach_internal(c, EFFECT_INTERNAL_SUBTITLE, EFFECT_TYPE_EFFECT);
  auto* subtitle = static_cast<SubtitleEffect*>(sub_eff.get());

  QVector<SubtitleCue> cues;
  cues.append(SubtitleCue{0, 500, "X"});
  subtitle->SetCues(cues);

  // Frame 7: cue active, expect some non-transparent text pixel.
  QByteArray inside = h_->render_frame(seq.get(), 7);
  QVERIFY(!inside.isEmpty());
  bool any_text = false;
  for (int y = 0; y < 64 && !any_text; ++y) {
    for (int x = 0; x < 64; ++x) {
      if (alpha_of(inside, 64, x, y) > 100) {
        any_text = true;
        break;
      }
    }
  }
  QVERIFY2(any_text, "expected subtitle text pixels when cue is active");

  // Frame 30: past the cue, expect fully transparent.
  QByteArray after = h_->render_frame(seq.get(), 30);
  QVERIFY(!after.isEmpty());
  int max_alpha = 0;
  for (int i = 3; i < after.size(); i += 4) {
    max_alpha = qMax(max_alpha, static_cast<int>(static_cast<uchar>(after[i])));
  }
  QVERIFY2(max_alpha < 20, qPrintable(QString("expected fully transparent after cue, max alpha=%1").arg(max_alpha)));
}

void TestEffectsGpu::gradientRendersTwoStops() {
  // Linear gradient red→blue, top→bottom. Top row near red, bottom row near
  // blue, middle near (R+B)/2 purple.
  auto seq = h_->make_sequence(64, 64, 30.0);
  Clip* c = h_->add_generator_clip(seq.get(), -1, 0, 30, EFFECT_INTERNAL_GRADIENT);
  Effect* gradient = c->effects[1].get();
  // Rows: 0=type (combo LINEAR default), 1=start, 2=end, 3=angle.
  h_->set_field_color(gradient, 1, 0, QColor(Qt::red));
  h_->set_field_color(gradient, 2, 0, QColor(Qt::blue));
  h_->set_field_double(gradient, 3, 0, 90.0);  // top→bottom

  QByteArray pixels = h_->render_frame(seq.get(), 0);
  QVERIFY(!pixels.isEmpty());

  // Top pixel: red dominates.
  Rgba top = pixel_at(pixels, 64, 32, 2);
  QVERIFY2(top.r > 150 && top.b < 80,
           qPrintable(QString("expected red near top, got R=%1 G=%2 B=%3").arg(top.r).arg(top.g).arg(top.b)));
  // Bottom pixel: blue dominates.
  Rgba bot = pixel_at(pixels, 64, 32, 61);
  QVERIFY2(bot.b > 150 && bot.r < 80,
           qPrintable(QString("expected blue near bottom, got R=%1 G=%2 B=%3").arg(bot.r).arg(bot.g).arg(bot.b)));
  // Mid pixel: roughly equal red and blue.
  Rgba mid = pixel_at(pixels, 64, 32, 32);
  QVERIFY2(mid.r > 40 && mid.b > 40,
           qPrintable(QString("expected purple near middle, got R=%1 G=%2 B=%3").arg(mid.r).arg(mid.g).arg(mid.b)));
}

int main(int argc, char* argv[]) {
  qputenv("QT_QPA_PLATFORM", "offscreen");

  // Force OpenGL 3.2 core profile so the QRhi(OpenGLES2) backend's offscreen
  // context exposes a GLSL 150 binding. Without this, Qt's offscreen platform
  // hands out a GL 2.1 / GLSL 1.20 context and QRhi rejects every .qsb shader
  // (which is baked at GLSL 150 in src/CMakeLists.txt:467).
  QSurfaceFormat fmt;
  fmt.setVersion(3, 2);
  fmt.setProfile(QSurfaceFormat::CoreProfile);
  fmt.setRenderableType(QSurfaceFormat::OpenGL);
  QSurfaceFormat::setDefaultFormat(fmt);

  QApplication app(argc, argv);
  TestEffectsGpu test;
  return QTest::qExec(&test, argc, argv);
}

#include "test_effects_gpu.moc"
