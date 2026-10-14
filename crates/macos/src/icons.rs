//! Menu bar icons, drawn at run time as template images so macOS tints them for light and dark bars.

use tray_icon::Icon;

const SIZE: u32 = 44;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Idle,
    Listening,
    Working,
    Error,
}

/// Coverage of a pixel center by a capsule of radius `r` around the segment from `(x, y0)` to `(x, y1)`.
fn capsule(px: f32, py: f32, x: f32, y0: f32, y1: f32, r: f32) -> f32 {
    let y = py.clamp(y0, y1);
    let d = ((px - x).powi(2) + (py - y).powi(2)).sqrt();
    (r + 0.5 - d).clamp(0.0, 1.0)
}

/// Coverage of a pixel center by an ellipse, with the soft edge measured along the shorter radius.
fn ellipse(px: f32, py: f32, cx: f32, cy: f32, rx: f32, ry: f32) -> f32 {
    let k = ((px - cx) / rx).hypot((py - cy) / ry);
    ((1.0 - k) * rx.min(ry) + 0.5).clamp(0.0, 1.0)
}

fn rect(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let h = ((px - x0).min(x1 - px) + 0.5).clamp(0.0, 1.0);
    let v = ((py - y0).min(y1 - py) + 0.5).clamp(0.0, 1.0);
    h * v
}

fn ring(px: f32, py: f32, cx: f32, cy: f32, r: f32, width: f32) -> f32 {
    let d = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
    (width / 2.0 + 0.5 - (d - r).abs()).clamp(0.0, 1.0)
}

/// A microphone in a peaked cap: hollow when idle, filled while listening, with a dot while working
/// or on error.
pub fn icon(state: State) -> Icon {
    let s = SIZE as f32 / 22.0;
    let mut rgba = vec![0u8; (SIZE * SIZE * 4) as usize];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            // The cap: crown, band, and the lower half of a visor, with a hole for the badge.
            let crown = ellipse(px, py, 11.0 * s, 3.6 * s, 6.2 * s, 2.2 * s);
            let band = rect(px, py, 7.0 * s, 4.6 * s, 15.0 * s, 6.3 * s);
            let visor = if py > 6.8 * s {
                ellipse(px, py, 11.0 * s, 6.8 * s, 5.0 * s, 1.7 * s)
            } else {
                0.0
            };
            let badge = ellipse(px, py, 11.0 * s, 4.4 * s, s, s);
            let cap = crown.max(band).max(visor) * (1.0 - badge);
            let body_outer = capsule(px, py, 11.0 * s, 11.9 * s, 13.2 * s, 2.6 * s);
            let body_inner = capsule(px, py, 11.0 * s, 11.9 * s, 13.2 * s, 1.2 * s);
            let body = if state == State::Listening {
                body_outer
            } else {
                body_outer * (1.0 - body_inner)
            };
            // The lower half of a ring around the body, and the stand.
            let cradle = if py > 13.2 * s {
                ring(px, py, 11.0 * s, 13.2 * s, 4.6 * s, 1.4 * s)
            } else {
                0.0
            };
            let stand = capsule(px, py, 11.0 * s, 18.2 * s, 20.8 * s, 0.75 * s);
            let dot = match state {
                State::Working | State::Error => {
                    capsule(px, py, 18.6 * s, 18.4 * s, 18.4 * s, 2.3 * s)
                }
                _ => 0.0,
            };
            let alpha = cap.max(body).max(cradle).max(stand).max(dot);
            let i = ((y * SIZE + x) * 4) as usize;
            rgba[i + 3] = (alpha * 255.0) as u8;
        }
    }
    Icon::from_rgba(rgba, SIZE, SIZE).expect("icon buffer matches its size")
}
