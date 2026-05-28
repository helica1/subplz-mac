#!/usr/bin/env bash
# Build SubPlz.app — a clickable Mac launcher for subplz_gui.py.
#
# Single-purpose AppleScript wrapper: when clicked, it backgrounds the GUI
# process if no instance is already running, then exits. No Terminal window.
#
# Rerun this script after you move the project, after Python venv changes
# location, or anytime you want to regenerate the bundle.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="$PROJECT_DIR/SubPlz.app"
SCRIPT_PATH="$(mktemp -t subplz_launcher).applescript"

cat >"$SCRIPT_PATH" <<APPLESCRIPT
on run
    set projectDir to "$PROJECT_DIR"
    set guiScript to projectDir & "/subplz_gui.py"
    set pythonBin to projectDir & "/.venv/bin/python"
    -- Skip launch if an instance is already running (matched by absolute path).
    try
        do shell script "pgrep -f " & quoted form of guiScript & " > /dev/null"
        return -- already running; just exit silently
    on error
        -- not running -> launch and detach
    end try
    do shell script "cd " & quoted form of projectDir & " && nohup " & quoted form of pythonBin & " " & quoted form of guiScript & " >/tmp/subplz_gui.log 2>&1 &"
end run
APPLESCRIPT

rm -rf "$APP_PATH"
osacompile -o "$APP_PATH" "$SCRIPT_PATH"
rm -f "$SCRIPT_PATH"

# Optional: custom icon. Drop a 1024x1024 PNG at build/SubPlz.png to use it.
ICON_SRC="$PROJECT_DIR/build/SubPlz.png"
if [[ -f "$ICON_SRC" ]]; then
    ICONSET="$(mktemp -d -t subplz_iconset)/SubPlz.iconset"
    mkdir -p "$ICONSET"
    for size in 16 32 64 128 256 512 1024; do
        sips -z "$size" "$size" "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
        # @2x retina sibling (skip the largest, no point in 2048×2048)
        if [[ "$size" -lt 1024 ]]; then
            sips -z "$((size*2))" "$((size*2))" "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
        fi
    done
    iconutil -c icns -o "$APP_PATH/Contents/Resources/applet.icns" "$ICONSET"
    rm -rf "$(dirname "$ICONSET")"
    # Touch to force Finder to refresh the cached icon
    touch "$APP_PATH"
fi

echo "Built $APP_PATH"
echo ""
echo "Try it: open '$APP_PATH'"
echo "Pin to Dock: drag the icon from Finder onto the Dock."
echo "Add to /Applications or Launchpad: drag it there."
echo ""
echo "Custom icon: drop a 1024×1024 PNG at build/SubPlz.png and rerun this script."
