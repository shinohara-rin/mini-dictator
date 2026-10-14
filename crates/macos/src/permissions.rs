//! The privacy permissions the app needs: the microphone to hear, and accessibility both to read
//! the screen and to paste. The menu polls these, because macOS sends no notice when the user
//! grants one in System Settings.

use objc2_av_foundation::{AVAuthorizationStatus, AVCaptureDevice, AVMediaTypeAudio};

use crate::context::is_trusted;

/// The microphone's authorization. AVMediaTypeAudio is always set on macOS; without it, there is
/// nothing to check.
fn microphone() -> AVAuthorizationStatus {
    // SAFETY: AVMediaTypeAudio is a constant string, and audio is a media type the method accepts.
    unsafe {
        match AVMediaTypeAudio {
            Some(audio) => AVCaptureDevice::authorizationStatusForMediaType(audio),
            None => AVAuthorizationStatus::Authorized,
        }
    }
}

/// The names of the permissions the app lacks. macOS asks for the microphone on the first
/// recording, so until then it is listed as not asked rather than as refused.
pub fn missing() -> Vec<&'static str> {
    let mut missing = Vec::new();
    match microphone() {
        AVAuthorizationStatus::Authorized => {}
        AVAuthorizationStatus::NotDetermined => missing.push("Microphone (asked on first use)"),
        _ => missing.push("Microphone"),
    }
    if !is_trusted(false) {
        missing.push("Accessibility");
    }
    missing
}

/// The menu line for the missing permissions.
pub fn summary(missing: &[&str]) -> String {
    if missing.is_empty() {
        "All permissions granted".to_owned()
    } else {
        format!("Needs access: {}", missing.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::summary;

    #[test]
    fn lists_what_is_missing() {
        assert_eq!(summary(&[]), "All permissions granted");
        assert_eq!(
            summary(&["Microphone", "Accessibility"]),
            "Needs access: Microphone, Accessibility"
        );
    }
}
