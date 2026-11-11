import 'package:flutter/material.dart';

import 'services/api_service.dart';
import 'screens/session_screen.dart';
import 'theme/app_theme.dart';

/// Backend base URL.
///
/// - Android emulator talking to a backend on your host machine: use
///   http://10.0.2.2:8000
/// - iOS simulator / desktop / web (same machine): use http://127.0.0.1:8000
/// - Physical tablet on the same network as the backend: use the host
///   machine's LAN IP, e.g. http://192.168.1.23:8000
///
/// Override at build/run time instead of editing this file:
///   flutter run --dart-define=BACKEND_URL=http://192.168.1.23:8000
const String kBackendUrl = String.fromEnvironment(
  'BACKEND_URL',
  defaultValue: 'http://127.0.0.1:8000',
);

void main() {
  runApp(const DysgraphiaTutorApp());
}

class DysgraphiaTutorApp extends StatelessWidget {
  final ApiService? api;

  const DysgraphiaTutorApp({super.key, this.api});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Dysgraphia Tutor',
      debugShowCheckedModeBanner: false,
      theme: buildAppTheme(),
      home: SessionScreen(api: api ?? ApiService(baseUrl: kBackendUrl)),
    );
  }
}
