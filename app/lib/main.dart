import 'package:flutter/material.dart';

import 'api/client.dart';
import 'screens/home_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ApiClient.init();
  runApp(const LongTennisApp());
}

class LongTennisApp extends StatelessWidget {
  const LongTennisApp({super.key});

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF1E4D8C);
    return MaterialApp(
      title: 'Long Tennis',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: seed), useMaterial3: true),
      darkTheme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: seed, brightness: Brightness.dark, primary: const Color(0xFFD7F036)),
          useMaterial3: true),
      home: const HomeScreen(),
    );
  }
}
