#!/usr/bin/env bash
# Ships the version in app/build.gradle.kts: builds the signed release APK, checks the signature,
# tags v<version>, pushes, and publishes the APK to GitHub Releases, where the site's download
# button (releases/latest) picks it up. Run from the repo root on a clean main.
#
#   tools/release.sh [--notes notes.md] [--dry-run]
#
# Signing comes from ~/.gradle/gradle.properties (board.keystore, board.keystorePass, board.keyPass)
# or the BOARD_KEYSTORE / BOARD_KEYSTORE_PASS / BOARD_KEY_PASS variables; without them the build
# is unsigned and this script stops before tagging anything.
set -euo pipefail

NOTES=""
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --notes) NOTES="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done

[ -f app/build.gradle.kts ] || { echo "run from the repo root"; exit 2; }
[ "$(git branch --show-current)" = "main" ] || { echo "release from main"; exit 2; }
[ -z "$(git status --porcelain)" ] || { echo "working tree not clean"; exit 2; }

VERSION=$(sed -n 's/.*versionName = "\([^"]*\)".*/\1/p' app/build.gradle.kts | head -1)
CODE=$(sed -n 's/.*versionCode = \([0-9]*\).*/\1/p' app/build.gradle.kts | head -1)
TAG="v$VERSION"
[ -n "$VERSION" ] || { echo "no versionName in app/build.gradle.kts"; exit 2; }
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then echo "$TAG already exists; bump the version first"; exit 2; fi

JBR="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
[ -d "$JBR" ] && export JAVA_HOME="$JBR"
export PATH="$JAVA_HOME/bin:$PATH"
BT=$(ls -d "$HOME"/Library/Android/sdk/build-tools/* | sort -V | tail -1)

echo "building Type $VERSION ($CODE)"
./gradlew -q assembleRelease
APK=app/build/outputs/apk/release/app-release.apk
[ -f "$APK" ] || { echo "no signed release APK (app-release-unsigned.apk means the signing properties are missing)"; exit 1; }

"$BT/apksigner" verify --print-certs "$APK" > /tmp/type-release-certs.txt
DIGEST=$(sed -n 's/.*SHA-256 digest: \([0-9a-f]*\).*/\1/p' /tmp/type-release-certs.txt | head -1)
DEBUG_DIGEST="f772681bf479e44dd59ce0c3d70523afedfb7e07499ea71992fccd2a71895ee1"
[ "$DIGEST" != "$DEBUG_DIGEST" ] || { echo "the APK is debug-signed; refusing to publish it"; exit 1; }
echo "signed by $DIGEST"

OUT="app/build/outputs/apk/release/type-$VERSION.apk"
cp "$APK" "$OUT"
SHA=$(shasum -a 256 "$OUT" | cut -d" " -f1)
echo "$SHA  type-$VERSION.apk" > "$OUT.sha256"

LAST=$(git describe --tags --abbrev=0 2>/dev/null || true)
if [ -z "$NOTES" ]; then
  NOTES=/tmp/type-release-notes.md
  {
    echo "Type $VERSION"
    echo
    if [ -n "$LAST" ]; then git log "$LAST..HEAD" --format='- %s' | grep -v "^- Merge"; else git log -20 --format='- %s'; fi
    echo
    echo "SHA-256: $SHA"
  } > "$NOTES"
fi

if [ "$DRY" = 1 ]; then echo "dry run: would tag $TAG and publish $OUT with notes from $NOTES"; cat "$NOTES"; exit 0; fi

git tag -a "$TAG" -m "Type $VERSION"
git push origin main "$TAG"
gh release create "$TAG" "$OUT" "$OUT.sha256" --title "Type $VERSION" --notes-file "$NOTES"
echo "published: $(gh release view "$TAG" --json url -q .url)"
