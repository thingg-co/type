#!/bin/sh
# Installs Type on the connected phone, makes it the active keyboard, and (for debug
# builds) pushes a model so it works immediately. Run from the repo root:
#   tools/sideload.sh                      # release APK, model downloads in-app
#   tools/sideload.sh debug path/to.gguf   # debug APK + push a local model file
set -e

ADB="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
VARIANT="${1:-release}"
MODEL="$2"
APK="app/build/outputs/apk/$VARIANT/app-$VARIANT.apk"

[ -f "$APK" ] || { echo "$APK not found; run ./gradlew assembleRelease (or assembleDebug) first"; exit 1; }

"$ADB" get-state >/dev/null 2>&1 || { echo "no device: enable USB debugging and accept the prompt"; exit 1; }
"$ADB" install -r "$APK"
"$ADB" shell ime enable co.thingg.type/com.aosmith.type.ime.TypeInputMethodService
"$ADB" shell ime set co.thingg.type/com.aosmith.type.ime.TypeInputMethodService

if [ -n "$MODEL" ]; then
    if [ "$VARIANT" = "debug" ]; then
        base="$(basename "$MODEL")"
        "$ADB" push "$MODEL" "/data/local/tmp/$base"
        "$ADB" shell "run-as co.thingg.type sh -c 'mkdir -p files/models && cp /data/local/tmp/$base files/models/'"
        "$ADB" shell "rm /data/local/tmp/$base"
        echo "model installed: $base"
    else
        echo "note: model push needs the debug build (run-as); use the in-app download instead"
    fi
fi

"$ADB" shell am start -n co.thingg.type/com.aosmith.type.SettingsActivity
echo "done - Type is the active keyboard"
