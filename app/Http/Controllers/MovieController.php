<?php

namespace App\Http\Controllers;

use Illuminate\Support\Facades\DB;

class MovieController extends Controller
{
    public function index()
    {
        // $process = new Process(['python3', 'recom.py']);
        // $process->setTimeout(240);
        // $process->run();

        // if (!$process->isSuccessful()) {
        //     throw new ProcessFailedException($process);
        // }

        // $value = $process->getOutput();
        $value = 0.5172719;

        $value = ((float) str_replace('[[', '', explode("]]", $value)[0]));
        $movies = DB::select("SELECT * FROM `movies`
        ORDER BY ABS( `results` - $value)
        LIMIT 5");
        $movies['movie_id'] = $movies[0]->imdb_id;
        $movies['movie_id'] = ((int) str_replace("", '', $movies['movie_id']));
        $movies['film_details'] = new \Imdb\Title($movies['movie_id']);
        $movies['film_title'] = $movies['film_details']->title();
        $movies['rating'] = $movies['film_details']->rating();
        $movies['photo'] = $movies['film_details']->photo();
        $movies['titles'] = $movies['film_details']->title();
        if ($movies['film_details']->trailers()) {
            $movies['video'] = $movies['film_details']->trailers()[0];
            $movies['response'] = file_get_contents($movies['video']);
            $movies['vide'] = (str_replace(':\"', '', explode('\"}]}]"]', explode('"video/mp4\",\"url\"', $movies['response'])[2])[0])); // '{"id": 1420053, "name": "guzzle", ...}'
        }


        return view('index', $movies);
    }
}
